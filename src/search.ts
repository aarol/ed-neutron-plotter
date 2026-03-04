import * as directionSvg from "./directions.svg";
import styles from "./search.module.css";

export interface SearchBoxOptions {
  onSearch?: (query: string) => void;
  onSuggest: (word: string) => string[];
  onClickRoute?: (word: string) => void;
  placeholder: string;
  className?: string;
}

export class SearchBox {
  private container: HTMLDivElement;
  private inputWrapper: HTMLDivElement;
  private input: HTMLInputElement;
  private suggestionsContainer: HTMLDivElement;
  private routeIcon: HTMLDivElement;
  private onSearchCallback?: (query: string) => void;
  private onSuggestCallback: (word: string) => string[];
  private onClickRouteCallback?: (word: string) => void;
  private selectedIndex: number = -1;
  private suggestions: string[] = [];

  constructor(options: SearchBoxOptions) {
    this.onSearchCallback = options.onSearch;
    this.onSuggestCallback = options.onSuggest;
    this.onClickRouteCallback = options.onClickRoute;

    // Create container
    this.container = document.createElement('div');
    this.container.className = `${styles.container} ${options.className || ''}`.trim();

    // Create input wrapper (so suggestions position relative to it)
    this.inputWrapper = document.createElement('div');
    this.inputWrapper.className = styles.inputWrapper;

    // Create input
    this.input = document.createElement('input');
    this.input.type = 'text';
    this.input.placeholder = options.placeholder;
    this.input.className = styles.input;

    // Create route icon
    this.routeIcon = document.createElement('div');
    this.routeIcon.className = styles.routeIcon;

    const iconImg = document.createElement('img');
    iconImg.src = directionSvg.default;
    this.routeIcon.appendChild(iconImg);
    this.routeIcon.title = 'Find route to target';

    // Route icon click handler
    this.routeIcon.addEventListener('click', () => {
      if (this.input.value.trim() && this.onClickRouteCallback) {
        this.onClickRouteCallback(this.input.value);
      }
    });

    // Create suggestions container
    this.suggestionsContainer = document.createElement('div');
    this.suggestionsContainer.className = styles.suggestions;

    this.setupEventListeners();

    this.inputWrapper.appendChild(this.input);
    this.inputWrapper.appendChild(this.routeIcon);
    this.inputWrapper.appendChild(this.suggestionsContainer);
    this.container.appendChild(this.inputWrapper);
  }

  private setupEventListeners(): void {
    // Input functionality
    this.input.addEventListener('input', (e) => {
      const query = (e.target as HTMLInputElement).value;

      // Show/hide route icon based on input
      if (query.trim() && this.onClickRouteCallback) {
        this.routeIcon.classList.add(styles.routeIconVisible);
      } else {
        this.routeIcon.classList.remove(styles.routeIconVisible);
      }

      if (query.length >= 2) {
        const suggestions = this.onSuggestCallback(query);
        this.showSuggestions(suggestions);
      } else {
        this.hideSuggestions();
      }
    });

    // Keyboard navigation
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
      item.className = styles.suggestionItem;
      item.textContent = suggestion;
      item.dataset.index = index.toString();

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

    this.suggestionsContainer.classList.add(styles.suggestionsVisible);
  }

  private updateSelection(): void {
    const items = this.suggestionsContainer.querySelectorAll(`.${styles.suggestionItem}`);
    items.forEach((item, index) => {
      const element = item as HTMLElement;
      if (index === this.selectedIndex) {
        element.classList.add(styles.selected);
        element.scrollIntoView({ block: 'nearest' });
      } else {
        element.classList.remove(styles.selected);
      }
    });
  }

  private hideSuggestions(): void {
    this.suggestionsContainer.classList.remove(styles.suggestionsVisible);
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

    // Update route icon visibility when setting value programmatically
    if (value.trim() && this.onClickRouteCallback) {
      this.routeIcon.classList.add(styles.routeIconVisible);
    } else {
      this.routeIcon.classList.remove(styles.routeIconVisible);
    }
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
