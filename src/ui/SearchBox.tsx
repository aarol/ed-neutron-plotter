import * as directionSvg from "../directions.svg";
import { forwardRef } from "preact/compat";
import type { ComponentChildren } from "preact";
import { useEffect, useImperativeHandle, useRef, useState } from "preact/hooks";
import { Button } from "./components/Button";
import { uiTheme } from "./theme";

export interface SearchSubmitOptions {
  openRoute: boolean;
}

export interface SearchBoxProps {
  onSearch?: (query: string, options: SearchSubmitOptions) => void;
  onSuggest: (word: string) => string[];
  onClickRoute?: (word: string) => void;
  rightIcon?: ComponentChildren;
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
    rightIcon,
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
  const hasRightIcon = rightIcon !== undefined && rightIcon !== null;
  const hasRightControls = hasRightIcon || isRouteVisible;

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

  const submitSearch = (query: string, options: SearchSubmitOptions = { openRoute: false }) => {
    onSearch?.(query, options);
    hideSuggestions();
  };

  const selectSuggestion = (suggestion: string, options: SearchSubmitOptions = { openRoute: false }) => {
    setCurrentValue(suggestion);
    submitSearch(suggestion, options);
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
          className={`${uiTheme.textInput} w-100 min-w-20 ${hasRightIcon ? "pr-14" : hasRightControls ? "pr-20" : ""} ${inputClassName || ""}`.trim()}
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
              const options = { openRoute: event.shiftKey };

              if (selectedIndex >= 0 && selectedIndex < suggestions.length) {
                selectSuggestion(suggestions[selectedIndex], options);
                return;
              }

              submitSearch(value, options);
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

        <div className={`absolute inset-y-0 right-0 flex items-center ${!hasRightControls ? "hidden" : ""}`.trim()}>
          {hasRightIcon ? (
            <div className="flex h-full w-12 items-center justify-center border-l border-white/30">
              {rightIcon}
            </div>
          ) : null}

          {!hasRightIcon && isRouteVisible ? (
            <Button
              aria-label="Find route to target"
              className="mr-2 flex h-6 w-6 items-center justify-center border border-transparent bg-transparent p-0 opacity-70 hover:opacity-100"
              onClick={() => {
                if (value.trim() && onClickRoute) {
                  onClickRoute(value);
                }
              }}
              title="Find route to target"
              variant="plain"
            >
              <img alt="" aria-hidden="true" className="h-5 w-5 invert" src={directionSvg.default} />
            </Button>
          ) : null}
        </div>

        <div className={`${uiTheme.suggestionsPanel} ${suggestions.length > 0 ? "block" : "hidden"}`.trim()}>
          {suggestions.map((suggestion, index) => (
            <div
              className={`${uiTheme.suggestionRow} ${index === selectedIndex ? "bg-white/20" : ""}`.trim()}
              key={`${suggestion}-${index}`}
              onClick={() => selectSuggestion(suggestion, { openRoute: false })}
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