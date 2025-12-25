import { SearchBox, type SearchBoxOptions } from "./search";
import styles from "./route-dialog.module.css";

export interface RouteDialogOptions {
  onSuggest: (word: string) => string[];
  onRouteGenerated?: (config: RouteConfig) => void;
}

export interface RouteConfig {
  from: string;
  to: string;
  alreadySupercharged: boolean;
}

// Simple DOM creation helper
function createElement<T extends keyof HTMLElementTagNameMap>(
  tag: T,
  options: {
    className?: string;
    attributes?: Record<string, string>;
    children?: (HTMLElement | string | Text)[];
  } = {}
): HTMLElementTagNameMap[T] {
  const element = document.createElement(tag);

  if (options.className) {
    element.className = options.className;
  }

  if (options.attributes) {
    Object.entries(options.attributes).forEach(([key, value]) => {
      element.setAttribute(key, value);
    });
  }

  if (options.children) {
    options.children.forEach(child => {
      if (typeof child === 'string') {
        element.appendChild(document.createTextNode(child));
      } else {
        element.appendChild(child);
      }
    });
  }

  return element;
}

export class RouteDialog {
  private panel: HTMLDivElement;
  private fromSearchBox!: SearchBox;
  private toSearchBox!: SearchBox;
  private superchargedCheckbox!: HTMLInputElement;
  private onRouteGeneratedCallback?: (config: RouteConfig) => void;
  private isVisible: boolean = false;

  private options: RouteDialogOptions;

  constructor(options: RouteDialogOptions) {
    this.options = options;
    this.panel = this.createPanel();
    this.setupEventListeners();
    // Mount immediately so it's always ready
    document.body.appendChild(this.panel);
  }

  private createPanel(): HTMLDivElement {
    // Create panel element
    const panel = createElement('div', {
      className: styles.panel
    });

    // Create header
    const header = createElement('div', {
      className: styles.header
    });

    const title = createElement('h2', {
      children: ['Configure Route'],
      className: styles.title
    });

    const closeBtn = createElement('button', {
      className: styles.closeBtn,
      children: ['×'],
      attributes: { type: 'button' }
    });

    header.appendChild(title);
    header.appendChild(closeBtn);

    // Create form sections
    const fromSection = this.createFormSection('From:', 'from-container');
    const toSection = this.createFormSection('To:', 'to-container');

    // Create checkbox section
    const checkboxSection = createElement('div', {
      className: styles.checkboxSection
    });

    this.superchargedCheckbox = createElement('input', {
      attributes: { type: 'checkbox' },
      className: styles.checkbox
    });

    const checkboxLabel = createElement('label', {
      className: styles.checkboxLabel,
      children: [
        this.superchargedCheckbox,
        document.createTextNode('Already supercharged')
      ]
    });

    checkboxSection.appendChild(checkboxLabel);

    // Create buttons
    const buttonGroup = createElement('div', {
      className: styles.buttonGroup
    });

    const cancelBtn = createElement('button', {
      className: styles.cancelBtn,
      children: ['Cancel'],
      attributes: { type: 'button' }
    });

    const generateBtn = createElement('button', {
      className: styles.generateBtn,
      children: ['Generate Route'],
      attributes: { type: 'button' }
    });

    buttonGroup.appendChild(cancelBtn);
    buttonGroup.appendChild(generateBtn);

    // Assemble panel
    panel.appendChild(header);
    panel.appendChild(fromSection);
    panel.appendChild(toSection);
    panel.appendChild(checkboxSection);
    panel.appendChild(buttonGroup);

    // Create SearchBox instances
    this.createSearchBoxes(fromSection, toSection);

    return panel;
  }

  private createFormSection(labelText: string, className: string): HTMLDivElement {
    const section = createElement('div', {
      className: `${styles.formSection} ${className}`
    });

    const label = createElement('label', {
      children: [labelText],
      className: styles.label
    });

    section.appendChild(label);
    return section;
  }

  private createSearchBoxes(fromSection: HTMLDivElement, toSection: HTMLDivElement): void {
    const searchBoxOptions: Omit<SearchBoxOptions, 'placeholder'> = {
      onSuggest: this.options.onSuggest,
      onClickRoute: undefined,
      className: styles.dialogSearchBox
    };

    this.fromSearchBox = new SearchBox({
      ...searchBoxOptions,
      placeholder: 'Enter starting star...'
    });

    this.toSearchBox = new SearchBox({
      ...searchBoxOptions,
      placeholder: 'Enter destination star...'
    });

    this.fromSearchBox.mount(fromSection);
    this.toSearchBox.mount(toSection);
  }

  private setupEventListeners(): void {
    const closeBtn = this.panel.querySelector(`.${styles.closeBtn}`) as HTMLButtonElement;
    const cancelBtn = this.panel.querySelector(`.${styles.cancelBtn}`) as HTMLButtonElement;
    const generateBtn = this.panel.querySelector(`.${styles.generateBtn}`) as HTMLButtonElement;

    // Button click handlers
    closeBtn.addEventListener('click', () => this.close());
    cancelBtn.addEventListener('click', () => this.close());
    generateBtn.addEventListener('click', () => this.handleGenerate());

    // Escape key (global listener)
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && this.isVisible) {
        this.close();
      }
    });
  }

  private async handleGenerate(): Promise<void> {

    const from = this.fromSearchBox.getValue().trim();
    const to = this.toSearchBox.getValue().trim();


    const config: RouteConfig = {
      from, to, alreadySupercharged: this.superchargedCheckbox.checked
    };

    if (config.from && config.to) {
      if (this.onRouteGeneratedCallback) {
        this.onRouteGeneratedCallback(config);
      }
      this.close();
    }
  }

  public open(): Promise<RouteConfig | null> {
    return new Promise((resolve) => {
      // Set up one-time callback
      this.onRouteGeneratedCallback = (config: RouteConfig) => {
        resolve(config);
      };

      // Show panel
      this.panel.classList.add(styles.panelVisible);
      this.isVisible = true;

      // Focus first input
      setTimeout(() => {
        this.fromSearchBox.focus();
      }, 100);
    });
  }

  public close(): void {
    this.panel.classList.remove(styles.panelVisible);
    this.isVisible = false;
  }

  public destroy(): void {
    this.fromSearchBox.unmount();
    this.toSearchBox.unmount();
    if (this.panel.parentNode) {
      this.panel.parentNode.removeChild(this.panel);
    }
  }

  public setFromValue(value: string): void {
    this.fromSearchBox.setValue(value);
  }

  public setToValue(value: string): void {
    this.toSearchBox.setValue(value);
  }

  public setSuperchargedValue(value: boolean): void {
    this.superchargedCheckbox.checked = value;
  }
}
