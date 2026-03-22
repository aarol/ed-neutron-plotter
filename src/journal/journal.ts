import type { StarSystem } from "../ui/types";

type LocationCallback = (starSystem: StarSystem) => void;

export class Journal {
  private directoryHandle: FileSystemDirectoryHandle | null = null;
  private lastLocation: string | null = null;
  private observer: FileSystemObserver | null = null;

  onNewLocation: LocationCallback;

  constructor({ onNewLocation }: { onNewLocation: LocationCallback }) {
    this.onNewLocation = onNewLocation;
  }

  async init(): Promise<void> {
    if ('showDirectoryPicker' in window) {
      const handle = await window.showDirectoryPicker({ mode: "read" });
      this.directoryHandle = handle;
      await this.initObserver(handle);
      const latestFile = await this.findLatestJournalFile();
      if (latestFile) {
        const events = await this.parseJournalFile(latestFile);
        const lastJump = events.at(-1);
        if (lastJump) this.update(lastJump);
      }
    } else {
      throw new Error("Local filesystem access is not supported in this browser. Please use a compatible browser (e.g. Chrome, Edge) to access the journal.");
    }
    return;
  }

  async initObserver(handle: FileSystemDirectoryHandle): Promise<void> {
    this.stopTracking();

    const observer = new FileSystemObserver((record, _observer) => {
      record.forEach(async change => {
        if (change.changedHandle.kind === "file" && change.changedHandle.name.startsWith("Journal.")) {
          const events = await this.parseJournalFile(await change.changedHandle.getFile());
          const lastJump = events.at(-1);
          if (lastJump) this.update(lastJump);
        }
      })
    });

    await observer.observe(handle);
    this.observer = observer;
  }

  stopTracking(): void {
    if (!this.observer) {
      return;
    }

    if (this.directoryHandle) {
      try {
        this.observer.unobserve(this.directoryHandle);
      } catch {
        // Ignore cleanup errors if observer was already detached.
      }
    }
    this.observer.disconnect();
    this.observer = null;
    this.lastLocation = null;
  }

  get tracking(): boolean {
    return this.observer !== null;
  }

  async findLatestJournalFile(): Promise<File | null> {
    const journalFiles: File[] = [];
    for await (const [name, handle] of this.directoryHandle!.entries()) {
      if (handle.kind == "file" && name.startsWith("Journal.")) {
        journalFiles.push(await handle.getFile());
      }
    }
    journalFiles.sort((a, b) => a.lastModified - b.lastModified); // Assume that the latest journal file is the one with the most recent lastModified timestamp
    return journalFiles.at(-1) || null; // Return the latest journal file or null if none found
  }

  async parseJournalFile(file: File): Promise<FSDJumpEvent[]> {
    const text = await file.text();
    const events: FSDJumpEvent[] = [];

    for (const line of text.split("\n").filter(l => l.length > 0)) {
      const entry: JournalEntry = JSON.parse(line);
      if (entry.event === "FSDJump") {
        events.push(entry as FSDJumpEvent);
      }
    }
    return events;
  }

  update(event: FSDJumpEvent): void {
    if (event.StarSystem !== this.lastLocation) {
      this.lastLocation = event.StarSystem;
      this.onNewLocation({
        name: event.StarSystem,
        coords: {
          x: event.StarPos[0] / 1000.0,
          y: event.StarPos[1] / 1000.0,
          z: event.StarPos[2] / 1000.0,
        }
      });
    }
  }
}

type JournalEntry = {
  event: string;
};

interface FSDJumpEvent {
  timestamp: string
  event: "FSDJump"
  StarSystem: string
  SystemAddress: number
  StarPos: number[]
  JumpDist: number
  FuelUsed: number
  FuelLevel: number
}
