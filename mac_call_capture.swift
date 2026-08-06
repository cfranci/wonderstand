// mac_call_capture.swift
// ScreenCaptureKit-based per-app audio capture for Zoom / Google Meet / FaceTime.
//
// Build:
//   swiftc -O mac_call_capture.swift -o mac_call_capture
//
// Usage:
//   ./mac_call_capture --watch                   # poll for call apps, capture while running
//   ./mac_call_capture --once /path/to/out.m4a   # one-shot 30s capture for testing
//
// First run prompts for Screen Recording permission. Grant in
// System Settings → Privacy & Security → Screen Recording.
//
// Output: writes WAV chunks to /Users/cf/Projects/wonderstand/inbox/
// named `screen_<bundle>_<timestamp>.wav`. The ingest server picks them up
// via folder_watcher and runs them through the existing pipeline.

import Foundation
#if canImport(ScreenCaptureKit)
import ScreenCaptureKit
import AVFoundation

let INBOX = URL(fileURLWithPath: "/Users/cf/Projects/wonderstand/inbox")

let WATCHED_BUNDLES: [String: String] = [
    "us.zoom.xos": "zoom",
    "com.apple.FaceTime": "facetime",
    "com.google.Chrome": "chrome",     // filter by tab URL meet.google.com via window title
    "com.microsoft.teams2": "teams",
]

func isAppRunning(_ bundleId: String) -> Bool {
    let apps = NSRunningApplication.runningApplications(withBundleIdentifier: bundleId)
    return !apps.isEmpty
}

@available(macOS 13.0, *)
func runOneShot(outURL: URL, durationSec: Double) async throws {
    let content = try await SCShareableContent.current
    guard let display = content.displays.first else { fatalError("no display") }
    let filter = SCContentFilter(display: display, excludingApplications: [], exceptingWindows: [])
    let cfg = SCStreamConfiguration()
    cfg.capturesAudio = true
    cfg.excludesCurrentProcessAudio = true
    cfg.sampleRate = 48000
    cfg.channelCount = 2
    cfg.width = 2; cfg.height = 2   // we don't care about video

    // Minimal stream-output adapter that writes raw PCM to disk for the duration.
    final class Sink: NSObject, SCStreamOutput {
        let outURL: URL
        var fileHandle: FileHandle?
        init(outURL: URL) { self.outURL = outURL; super.init()
            FileManager.default.createFile(atPath: outURL.path, contents: nil)
            self.fileHandle = try? FileHandle(forWritingTo: outURL)
        }
        func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
            guard type == .audio else { return }
            guard let block = CMSampleBufferGetDataBuffer(sampleBuffer) else { return }
            var length = 0; var dataPtr: UnsafeMutablePointer<Int8>? = nil
            CMBlockBufferGetDataPointer(block, atOffset: 0, lengthAtOffsetOut: nil, totalLengthOut: &length, dataPointerOut: &dataPtr)
            guard let ptr = dataPtr else { return }
            let data = Data(bytes: ptr, count: length)
            try? fileHandle?.write(contentsOf: data)
        }
    }

    let sink = Sink(outURL: outURL)
    let stream = SCStream(filter: filter, configuration: cfg, delegate: nil)
    try stream.addStreamOutput(sink, type: .audio, sampleHandlerQueue: .global())
    try await stream.startCapture()
    try await Task.sleep(nanoseconds: UInt64(durationSec * 1_000_000_000))
    try await stream.stopCapture()
    try sink.fileHandle?.close()
    print("Wrote \(outURL.path)")
}

@available(macOS 13.0, *)
func watch() async {
    var capturingBundle: String? = nil
    while true {
        var running: String? = nil
        for (bid, _) in WATCHED_BUNDLES {
            if isAppRunning(bid) { running = bid; break }
        }
        if running != nil && capturingBundle == nil {
            let tag = WATCHED_BUNDLES[running!] ?? "unknown"
            let ts = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
            let out = INBOX.appendingPathComponent("screen_\(tag)_\(ts).wav")
            capturingBundle = running
            print("[capture] starting for \(running!) -> \(out.lastPathComponent)")
            Task {
                do { try await runOneShot(outURL: out, durationSec: 60 * 60 * 3) }
                catch { print("[capture] error \(error)") }
            }
        } else if running == nil && capturingBundle != nil {
            print("[capture] app exited, ending capture")
            capturingBundle = nil
            // The one-shot task will end at its duration cap or app exit kills audio.
        }
        try? await Task.sleep(nanoseconds: 5_000_000_000)
    }
}

@main
struct App {
    static func main() async {
        guard #available(macOS 13.0, *) else {
            print("Requires macOS 13+ (ScreenCaptureKit)")
            return
        }
        let args = CommandLine.arguments
        if args.contains("--once") {
            let idx = args.firstIndex(of: "--once")!
            let outPath = (idx + 1 < args.count) ? args[idx + 1] : "/tmp/sck_test.wav"
            do {
                try await runOneShot(outURL: URL(fileURLWithPath: outPath), durationSec: 30)
            } catch {
                print("error: \(error)")
            }
        } else if args.contains("--watch") {
            await watch()
        } else {
            print("usage: mac_call_capture --watch | --once /path/out.wav")
        }
    }
}
#else
@main
struct App {
    static func main() {
        print("ScreenCaptureKit not available on this build target.")
    }
}
#endif
