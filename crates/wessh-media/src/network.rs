//! Finding cameras on the local network: IP cameras and phone camera apps (IP Webcam, DroidCam,
//! RTSP servers) on the same Wi-Fi, so they can be picked instead of typed.
//!
//! Every address of this machine's /24 is tried at once, on the ports those cameras listen on;
//! an open port only counts once the server behind it answers like a camera (an RTSP reply, or a
//! phone app's web page), so a router's or a development server's web page isn't listed. Only a
//! private network is scanned. A scan takes about [`CONNECT_TIMEOUT`] per port.
//!
//! Classifying an answer is kept apart from the sockets so it is unit-tested.

use crate::VideoDevice;
use std::io::{Read, Write};
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, TcpStream, UdpSocket};
use std::time::Duration;

/// How long a host may take to accept a connection. Hosts on the same Wi-Fi answer in a few
/// milliseconds (closed ports refuse at once); absent ones never do.
const CONNECT_TIMEOUT: Duration = Duration::from_millis(400);
/// How long an open port may take to answer the probe.
const ANSWER_TIMEOUT: Duration = Duration::from_millis(800);
/// The start of an answer is enough to tell a camera; phone apps put their name in the page title.
const ANSWER_BYTES: usize = 4096;
/// Any routable address: connecting a UDP socket to it only picks the interface the default
/// route leaves through (no packet is sent), whose address is this machine's on the LAN.
const ROUTE_PROBE: SocketAddrV4 = SocketAddrV4::new(Ipv4Addr::new(192, 0, 2, 1), 9);

/// A kind of network camera, by the port it serves on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// An IP camera or a phone's RTSP server app.
    Rtsp,
    /// IP Webcam (Android), whose stream is `/video`.
    IpWebcam,
    /// DroidCam, whose stream is `/video` too.
    DroidCam,
}

/// The ports tried on every host and what is expected there.
const PORTS: [(u16, Kind); 4] = [(554, Kind::Rtsp), (8554, Kind::Rtsp), (8080, Kind::IpWebcam), (4747, Kind::DroidCam)];

impl Kind {
    /// What is sent once connected.
    fn probe(self, address: SocketAddrV4) -> String {
        match self {
            Self::Rtsp => format!("OPTIONS rtsp://{address}/ RTSP/1.0\r\nCSeq: 1\r\n\r\n"),
            Self::IpWebcam | Self::DroidCam => format!("GET / HTTP/1.0\r\nHost: {address}\r\n\r\n"),
        }
    }

    /// Whether `answer` (the start of the reply to [`Kind::probe`]) comes from this kind of camera.
    fn recognises(self, answer: &[u8]) -> bool {
        let text = String::from_utf8_lossy(answer).to_ascii_lowercase();
        match self {
            Self::Rtsp => text.starts_with("rtsp/1.0"),
            Self::IpWebcam => text.starts_with("http/") && text.contains("ip webcam"),
            Self::DroidCam => text.starts_with("http/") && text.contains("droidcam"),
        }
    }

    /// The camera as the device list shows it; its id is what `CameraConfig::device` takes.
    fn device(self, address: SocketAddrV4) -> VideoDevice {
        let (id, name) = match self {
            // The stream path differs between cameras; this opens the ones serving the root and
            // shows where to add a path or credentials.
            Self::Rtsp => (format!("rtsp://{address}/"), "RTSP camera"),
            // A bare phone-app address opens its `/video` stream.
            Self::IpWebcam => (format!("http://{address}/"), "IP Webcam"),
            Self::DroidCam => (format!("http://{address}/"), "DroidCam"),
        };
        VideoDevice { id, name: format!("{name} · {}", address.ip()) }
    }
}

/// Cameras answering on this machine's local network, by address. Blocking, for about a second
/// per port in [`PORTS`] at most; empty when offline or not on a private network.
#[must_use]
pub fn find_network_cameras() -> Vec<VideoDevice> {
    let Some(local) = local_address() else { return Vec::new() };
    let mut found: Vec<(Ipv4Addr, u16, VideoDevice)> = std::thread::scope(|scope| {
        let hosts: Vec<_> = neighbours(local).map(|host| scope.spawn(move || probe_host(host))).collect();
        hosts.into_iter().filter_map(|host| host.join().ok()).flatten().collect()
    });
    found.sort_by_key(|(host, port, _)| (*host, *port));
    found.into_iter().map(|(_, _, device)| device).collect()
}

/// This machine's address on the network the default route leaves through, when that network
/// is a private one (scanning a public /24 would knock on strangers' doors).
fn local_address() -> Option<Ipv4Addr> {
    let socket = UdpSocket::bind((Ipv4Addr::UNSPECIFIED, 0)).ok()?;
    socket.connect(ROUTE_PROBE).ok()?;
    match socket.local_addr().ok()? {
        SocketAddr::V4(address) if address.ip().is_private() => Some(*address.ip()),
        _ => None,
    }
}

/// Every host address of `local`'s /24, `local` included (a camera app may run on this machine).
fn neighbours(local: Ipv4Addr) -> impl Iterator<Item = Ipv4Addr> {
    let [a, b, c, _] = local.octets();
    (1..=254).map(move |d| Ipv4Addr::new(a, b, c, d))
}

/// The cameras `host` serves, one per answering port.
fn probe_host(host: Ipv4Addr) -> Vec<(Ipv4Addr, u16, VideoDevice)> {
    PORTS
        .iter()
        .filter(|&&(port, kind)| answers_as(SocketAddrV4::new(host, port), kind))
        .map(|&(port, kind)| (host, port, kind.device(SocketAddrV4::new(host, port))))
        .collect()
}

/// Whether a `kind` camera answers at `address`.
fn answers_as(address: SocketAddrV4, kind: Kind) -> bool {
    let Ok(mut stream) = TcpStream::connect_timeout(&address.into(), CONNECT_TIMEOUT) else { return false };
    if stream.set_read_timeout(Some(ANSWER_TIMEOUT)).is_err() || stream.write_all(kind.probe(address).as_bytes()).is_err() {
        return false;
    }
    let mut answer = Vec::with_capacity(ANSWER_BYTES);
    let _ = stream.take(ANSWER_BYTES as u64).read_to_end(&mut answer);
    kind.recognises(&answer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cameras_are_told_from_other_servers() {
        let rtsp = b"RTSP/1.0 200 OK\r\nCSeq: 1\r\nPublic: DESCRIBE, SETUP, PLAY\r\n\r\n";
        assert!(Kind::Rtsp.recognises(rtsp));
        assert!(!Kind::Rtsp.recognises(b"HTTP/1.1 400 Bad Request\r\n\r\n"));
        let phone = b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n<html><head><title>IP Webcam</title>";
        assert!(Kind::IpWebcam.recognises(phone));
        let router = b"HTTP/1.1 200 OK\r\n\r\n<html><head><title>Router login</title>";
        assert!(!Kind::IpWebcam.recognises(router));
        assert!(Kind::DroidCam.recognises(b"HTTP/1.0 200 OK\r\n\r\nDroidCam Wi-Fi server"));
        assert!(!Kind::DroidCam.recognises(b""));
    }

    #[test]
    fn found_cameras_open_their_stream() {
        let phone = Kind::IpWebcam.device(SocketAddrV4::new(Ipv4Addr::new(192, 168, 1, 20), 8080));
        assert_eq!(phone, VideoDevice { id: "http://192.168.1.20:8080/".to_owned(), name: "IP Webcam · 192.168.1.20".to_owned() });
        assert!(crate::is_network_source(&phone.id));
        let camera = Kind::Rtsp.device(SocketAddrV4::new(Ipv4Addr::new(10, 0, 0, 7), 554));
        assert_eq!(camera.id, "rtsp://10.0.0.7:554/");
    }

    #[test]
    fn the_whole_subnet_is_tried() {
        let hosts: Vec<_> = neighbours(Ipv4Addr::new(192, 168, 1, 37)).collect();
        assert_eq!(hosts.len(), 254);
        assert_eq!(hosts[0], Ipv4Addr::new(192, 168, 1, 1));
        assert_eq!(hosts[253], Ipv4Addr::new(192, 168, 1, 254));
    }
}
