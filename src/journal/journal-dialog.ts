import type { Journal } from "./journal";
import styles from "./journal-dialog.module.css";

export class JournalDialog {
  private dialog: HTMLDialogElement;

  private journal: Journal;

  constructor(journal: Journal) {
    this.journal = journal;
    this.dialog = document.createElement("dialog");
    this.dialog.id = "journalDialog";
    this.dialog.className = styles.dialog;

    // Header
    const header = document.createElement("div");
    header.className = styles.header;

    const title = document.createElement("h3");
    title.className = styles.title;
    title.textContent = "Elite Dangerous Journal";

    const closeBtn = document.createElement("button");
    closeBtn.className = styles.closeBtn;
    closeBtn.innerHTML = "&#x2715;";
    closeBtn.title = "Close";
    closeBtn.addEventListener("click", () => this.dialog.close());

    header.appendChild(title);
    header.appendChild(closeBtn);

    // Body
    const body = document.createElement("div");
    body.className = styles.body;

    const instruction = document.createElement("p");
    instruction.className = styles.instruction;
    const pathCode = document.createElement("span");
    pathCode.className = styles.path;
    pathCode.textContent =
      "C:\\Users\\<Username>\\Saved Games\\Frontier Developments\\Elite Dangerous";
    instruction.appendChild(
      document.createTextNode("Navigate to ")
    );
    instruction.appendChild(pathCode);
    instruction.appendChild(
      document.createTextNode(" and select the directory.")
    );

    const selectBtn = document.createElement("button");
    selectBtn.className = styles.selectBtn;
    selectBtn.textContent = "Select Directory";
    selectBtn.addEventListener("click", async () => {
      try {
        await this.journal.init()
        this.dialog.close();
      } catch (err) {
        
        console.error("Error accessing journal directory:", err);
      }
    });

    body.appendChild(instruction);
    body.appendChild(selectBtn);

    this.dialog.appendChild(header);
    this.dialog.appendChild(body);
  }

  public mount(parent: HTMLElement = document.body): void {
    parent.appendChild(this.dialog);
  }
}
