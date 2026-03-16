import * as directionSvg from "../directions.svg";
import { forwardRef } from "preact/compat";
import { useEffect, useImperativeHandle, useRef, useState } from "preact/hooks";
import { uiTheme } from "./theme";

export interface SearchBoxProps {
  onSearch?: (query: string) => void;
  onSuggest: (word: string) => string[];
  onClickRoute?: (word: string) => void;
  onValueChange?: (value: string) => void;
  placeholder: string;
  className?: string;
  inputClassName?: string;
  value?: string;
  defaultValue?: string;
  autoFocus?: boolean;
  showRouteButton?: boolean;
}

export interface SearchBoxHandle {
  getValue: () => string;
  setValue: (value: string) => void;
  focus: () => void;
  blur: () => void;
}

export const SearchBox = forwardRef<SearchBoxHandle, SearchBoxProps>(function SearchBox(
  {
    autoFocus = false,
    className,
    defaultValue = "",
    inputClassName,
    onClickRoute,
    onSearch,
    onSuggest,
    onValueChange,
    placeholder,
    showRouteButton = Boolean(onClickRoute),
    value: controlledValue,
  },
  ref,
) {
  const containerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const suggestionRefs = useRef<(HTMLDivElement | null)[]>([]);
  const [internalValue, setInternalValue] = useState(defaultValue);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const value = controlledValue ?? internalValue;
  const isRouteVisible = showRouteButton && value.trim().length > 0;

  const hideSuggestions = () => {
    setSuggestions([]);
    setSelectedIndex(-1);
  };

  const updateSuggestions = (query: string) => {
    if (query.length >= 2) {
      setSuggestions(onSuggest(query));
      setSelectedIndex(-1);
      return;
    }

    hideSuggestions();
  };

  const setCurrentValue = (nextValue: string) => {
    if (controlledValue === undefined) {
      setInternalValue(nextValue);
    }

    onValueChange?.(nextValue);
    updateSuggestions(nextValue);
  };

  const submitSearch = (query: string) => {
    onSearch?.(query);
    hideSuggestions();
  };

  const selectSuggestion = (suggestion: string) => {
    setCurrentValue(suggestion);
    submitSearch(suggestion);
  };

  useImperativeHandle(
    ref,
    () => ({
      getValue: () => value,
      setValue: (nextValue: string) => {
        setCurrentValue(nextValue);
      },
      focus: () => inputRef.current?.focus(),
      blur: () => inputRef.current?.blur(),
    }),
    [value],
  );

  useEffect(() => {
    const handleDocumentClick = (event: MouseEvent) => {
      if (!containerRef.current?.contains(event.target as Node)) {
        hideSuggestions();
      }
    };

    document.addEventListener("click", handleDocumentClick);
    return () => {
      document.removeEventListener("click", handleDocumentClick);
    };
  }, []);

  useEffect(() => {
    if (autoFocus) {
      inputRef.current?.focus();
    }
  }, [autoFocus]);

  useEffect(() => {
    if (selectedIndex >= 0) {
      suggestionRefs.current[selectedIndex]?.scrollIntoView({ block: "nearest" });
    }
  }, [selectedIndex]);

  return (
    <div className={`flex items-center gap-2 ${className || ""}`.trim()} ref={containerRef}>
      <div className="relative">
        <input
          className={`${uiTheme.textInput} w-[300px] ${inputClassName || ""}`.trim()}
          onInput={(event) => {
            const query = (event.currentTarget as HTMLInputElement).value;
            setCurrentValue(query);
          }}
          onKeyDown={(event) => {
            if (event.key === "ArrowDown") {
              event.preventDefault();
              if (suggestions.length > 0) {
                setSelectedIndex((currentIndex) => Math.min(currentIndex + 1, suggestions.length - 1));
              }
              return;
            }

            if (event.key === "ArrowUp") {
              event.preventDefault();
              if (suggestions.length > 0) {
                setSelectedIndex((currentIndex) => Math.max(currentIndex - 1, -1));
              }
              return;
            }

            if (event.key === "Enter") {
              event.preventDefault();
              if (selectedIndex >= 0 && selectedIndex < suggestions.length) {
                selectSuggestion(suggestions[selectedIndex]);
                return;
              }

              submitSearch(value);
              return;
            }

            if (event.key === "Escape") {
              hideSuggestions();
            }
          }}
          placeholder={placeholder}
          ref={inputRef}
          type="text"
          value={value}
        />

        <button
          aria-label="Find route to target"
          className={`absolute right-3 top-1/2 flex h-6 w-6 -translate-y-1/2 items-center justify-center border border-transparent bg-transparent p-0 opacity-70 transition hover:opacity-100 ${!isRouteVisible ? "hidden" : ""}`.trim()}
          onClick={() => {
            if (value.trim() && onClickRoute) {
              onClickRoute(value);
            }
          }}
          title="Find route to target"
          type="button"
        >
          <img alt="" aria-hidden="true" className="h-5 w-5 invert" src={directionSvg.default} />
        </button>

        <div className={`${uiTheme.suggestionsPanel} ${suggestions.length > 0 ? "block" : "hidden"}`.trim()}>
          {suggestions.map((suggestion, index) => (
            <div
              className={`${uiTheme.suggestionRow} ${index === selectedIndex ? "bg-white/20" : ""}`.trim()}
              key={`${suggestion}-${index}`}
              onClick={() => selectSuggestion(suggestion)}
              onMouseEnter={() => setSelectedIndex(index)}
              ref={(element) => {
                suggestionRefs.current[index] = element;
              }}
            >
              {suggestion}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
});