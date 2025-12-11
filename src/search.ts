export interface SearchBoxOptions {
  onSearch?: (query: string) => void;
  onSuggest: (word: string) => string[];
  placeholder?: string;
  className?: string;
}

export class SearchBox {
  private container: HTMLDivElement;
  private input: HTMLInputElement;
  private suggestionsContainer: HTMLDivElement;
  private onSearchCallback?: (query: string) => void;
  private onSuggestCallback: (word: string) => string[];
  private selectedIndex: number = -1;
  private suggestions: string[] = [];

  constructor(options: SearchBoxOptions) {
    this.onSearchCallback = options.onSearch;
    this.onSuggestCallback = options.onSuggest;
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

    // Create suggestions container
    this.suggestionsContainer = document.createElement('div');
    this.suggestionsContainer.className = 'search-suggestions';
    this.suggestionsContainer.style.cssText = `
      position: absolute;
      top: 100%;
      left: 0;
      right: 0;
      background: rgba(0, 0, 0, 0.9);
      border: 1px solid rgba(255, 255, 255, 0.3);
      border-radius: 15px 15px;
      max-height: 200px;
      overflow-y: auto;
      backdrop-filter: blur(10px);
      -webkit-backdrop-filter: blur(10px);
      display: none;
      z-index: 1001;
    `;

    // Add input functionality
    this.input.addEventListener('input', (e) => {
      const query = (e.target as HTMLInputElement).value;
      if (query.length >= 2) {
        const suggestions = this.onSuggestCallback(query);
        this.showSuggestions(suggestions);
      } else {
        this.hideSuggestions();
      }
    });

    this.input.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        if (this.suggestions.length > 0) {
          this.selectedIndex = Math.min(this.selectedIndex + 1, this.suggestions.length - 1);
          this.updateSelection();
        }
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        if (this.suggestions.length > 0) {
          this.selectedIndex = Math.max(this.selectedIndex - 1, -1);
          this.updateSelection();
        }
      } else if (e.key === 'Enter') {
        e.preventDefault();
        if (this.selectedIndex >= 0 && this.selectedIndex < this.suggestions.length) {
          const selectedSuggestion = this.suggestions[this.selectedIndex];
          this.input.value = selectedSuggestion;
          this.hideSuggestions();
          if (this.onSearchCallback) {
            this.onSearchCallback(selectedSuggestion);
          }
        } else {
          const query = this.input.value;
          if (this.onSearchCallback) {
            this.onSearchCallback(query);
          }
          this.hideSuggestions();
        }
      } else if (e.key === 'Escape') {
        this.hideSuggestions();
      }
    });

    // Hide suggestions when clicking outside
    document.addEventListener('click', (e) => {
      if (!this.container.contains(e.target as Node)) {
        this.hideSuggestions();
      }
    });

    this.container.appendChild(this.input);
    this.container.appendChild(this.suggestionsContainer);
  }

  private showSuggestions(suggestions: string[]): void {
    this.suggestions = suggestions;
    this.selectedIndex = -1;
    this.suggestionsContainer.innerHTML = '';

    if (suggestions.length === 0) {
      this.hideSuggestions();
      return;
    }

    suggestions.forEach((suggestion, index) => {
      const item = document.createElement('div');
      item.className = 'suggestion-item';
      item.textContent = suggestion;
      item.dataset.index = index.toString();
      item.style.cssText = `
        padding: 10px 20px;
        color: white;
        cursor: pointer;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        transition: background-color 0.2s ease;
      `;

      item.addEventListener('mouseenter', () => {
        this.selectedIndex = index;
        this.updateSelection();
      });

      item.addEventListener('click', () => {
        this.input.value = suggestion;
        this.hideSuggestions();
        if (this.onSearchCallback) {
          this.onSearchCallback(suggestion);
        }
      });

      this.suggestionsContainer.appendChild(item);
    });

    this.suggestionsContainer.style.display = 'block';
  }

  private updateSelection(): void {
    const items = this.suggestionsContainer.querySelectorAll('.suggestion-item');
    items.forEach((item, index) => {
      const element = item as HTMLElement;
      if (index === this.selectedIndex) {
        element.style.backgroundColor = 'rgba(255, 255, 255, 0.2)';
        element.scrollIntoView({ block: 'nearest' });
      } else {
        element.style.backgroundColor = 'transparent';
      }
    });
  }

  private hideSuggestions(): void {
    this.suggestionsContainer.style.display = 'none';
    this.selectedIndex = -1;
    this.suggestions = [];
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
