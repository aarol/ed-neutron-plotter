import type { ComponentChildren, ButtonHTMLAttributes } from "preact";

type ButtonVariant = "primary" | "secondary" | "icon" | "plain";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  children?: ComponentChildren;
  type?: "button" | "submit" | "reset";
  variant?: ButtonVariant;
}

const variantClassnames: Record<ButtonVariant, string> = {
  primary:
    "border border-space-accent-strong bg-space-accent-strong px-3 py-1.5 text-xs font-semibold text-white shadow-[0_2px_10px_rgba(47,127,243,0.5)] hover:bg-space-accent hover:shadow-[0_4px_16px_rgba(47,127,243,0.62)]",
  secondary:
    "border border-white/30 bg-transparent px-3 py-1.5 text-xs font-medium text-white/75 hover:border-white/50 hover:bg-white/10",
  icon:
    "flex h-8 w-8 items-center justify-center border border-white/25 bg-white/5 text-white/70 hover:bg-white/15 hover:text-white",
  plain: "",
};

const sharedClassnames = "cursor-pointer transition disabled:cursor-not-allowed disabled:opacity-60 disabled:shadow-none";

export function Button({ children, className, type = "button", variant = "plain", ...props }: ButtonProps) {
  const mergedClassnames = [sharedClassnames, variantClassnames[variant], className]
    .filter(Boolean)
    .join(" ");

  return (
    <button className={mergedClassnames} type={type} {...props}>
      {children}
    </button>
  );
}
