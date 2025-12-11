export interface SearchBoxOptions {
  onSearch?: (query: string) => void;
  onSuggest: (word: string) => string[];
  placeholder?: string;
  className?: string;
}

export class SearchBox {
  private container: HTMLDivElement;
  private input: HTMLInputElement;
  private onSearchCallback?: (query: string) => void;

  constructor(options: SearchBoxOptions) {
    this.onSearchCallback = options.onSearch;
    // Create container
    this.container = document.createElement('div');
    this.container.className = options.className || 'search-box-container';
    this.container.style.cssText = `
      position: fixed;
      top: 20px;
      left: 50%;
      transform: translateX(-50%);
      z-index: 1000;
      pointer-events: auto;
    `;

    // Create input
    this.input = document.createElement('input');
    this.input.type = 'text';
    this.input.placeholder = options.placeholder || 'Search stars...';
    this.input.className = 'search-box-input';
    this.input.style.cssText = `
      background: rgba(0, 0, 0, 0.7);
      border: 1px solid rgba(255, 255, 255, 0.3);
      border-radius: 25px;
      padding: 12px 20px;
      color: white;
      font-size: 16px;
      width: 300px;
      outline: none;
      backdrop-filter: blur(10px);
      -webkit-backdrop-filter: blur(10px);
      transition: all 0.3s ease;
      box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    `;

    // Add focus and blur effects
    this.input.addEventListener('focus', () => {
      this.input.style.background = 'rgba(0, 0, 0, 0.8)';
      this.input.style.borderColor = 'rgba(255, 255, 255, 0.6)';
      this.input.style.boxShadow = '0 6px 20px rgba(0, 0, 0, 0.3)';
    });

    this.input.addEventListener('blur', () => {
      this.input.style.background = 'rgba(0, 0, 0, 0.7)';
      this.input.style.borderColor = 'rgba(255, 255, 255, 0.3)';
      this.input.style.boxShadow = '0 4px 15px rgba(0, 0, 0, 0.2)';
    });

    // // Add search functionality
    // this.input.addEventListener('input', (e) => {
    //   const query = (e.target as HTMLInputElement).value;
    //   if (this.onSearchCallback) {
    //     this.onSearchCallback(query);
    //   }
    // });

    this.input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        const query = this.input.value;
        if (this.onSearchCallback) {
          this.onSearchCallback(query);
        }
      }
      if (this.input.value.length >= 2) {
        const query = this.input.value;
        const suggestions = options.onSuggest(query);
        console.log('Suggestions:', suggestions);
      }
    });

    this.container.appendChild(this.input);
  }

  public mount(parent: HTMLElement = document.body): void {
    parent.appendChild(this.container);
  }

  public unmount(): void {
    if (this.container.parentNode) {
      this.container.parentNode.removeChild(this.container);
    }
  }

  public getValue(): string {
    return this.input.value;
  }

  public setValue(value: string): void {
    this.input.value = value;
  }

  public focus(): void {
    this.input.focus();
  }

  public blur(): void {
    this.input.blur();
  }

  public setOnSearch(callback: (query: string) => void): void {
    this.onSearchCallback = callback;
  }
}
