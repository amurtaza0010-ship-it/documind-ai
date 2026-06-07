"""Design tokens for DocuMind AI design system."""

COLORS = {
    "bg_base": "#09090B",
    "bg_surface": "#18181B",
    "bg_elevated": "#27272A",
    "bg_glass": "rgba(24, 24, 27, 0.72)",
    "border_subtle": "rgba(255, 255, 255, 0.08)",
    "border_default": "rgba(255, 255, 255, 0.12)",
    "border_strong": "rgba(255, 255, 255, 0.18)",
    "text_primary": "#FAFAFA",
    "text_secondary": "#A1A1AA",
    "text_muted": "#71717A",
    "accent": "#818CF8",
    "accent_hover": "#A5B4FC",
    "accent_muted": "rgba(129, 140, 248, 0.15)",
    "success": "#34D399",
    "success_muted": "rgba(52, 211, 153, 0.12)",
    "warning": "#FBBF24",
    "warning_muted": "rgba(251, 191, 36, 0.12)",
    "error": "#F87171",
    "error_muted": "rgba(248, 113, 113, 0.12)",
    "user_bubble": "#27272A",
    "assistant_bubble": "transparent",
    "code_bg": "#1E1E2E",
    "code_border": "rgba(255, 255, 255, 0.1)",
}

SPACING = {
    "xs": "4px",
    "sm": "8px",
    "md": "12px",
    "lg": "16px",
    "xl": "24px",
    "2xl": "32px",
    "3xl": "48px",
}

RADIUS = {
    "sm": "6px",
    "md": "10px",
    "lg": "14px",
    "xl": "20px",
    "full": "9999px",
}

SHADOWS = {
    "sm": "0 1px 2px rgba(0, 0, 0, 0.4)",
    "md": "0 4px 12px rgba(0, 0, 0, 0.35)",
    "lg": "0 8px 32px rgba(0, 0, 0, 0.45)",
    "glow": "0 0 24px rgba(129, 140, 248, 0.15)",
}

TYPOGRAPHY = {
    "font_sans": "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    "font_mono": "'JetBrains Mono', 'Fira Code', 'Cascadia Code', Consolas, monospace",
    "text_xs": "0.75rem",
    "text_sm": "0.875rem",
    "text_base": "1rem",
    "text_lg": "1.125rem",
    "text_xl": "1.25rem",
    "text_2xl": "1.5rem",
    "text_3xl": "1.875rem",
    "leading_tight": "1.35",
    "leading_normal": "1.6",
    "leading_relaxed": "1.75",
    "tracking_normal": "0",
}

TRANSITIONS = {
    "fast": "150ms ease",
    "base": "200ms ease",
    "slow": "300ms ease",
}

SUGGESTED_PROMPTS = [
    "Summarize the key points of this document",
    "What are the main findings or conclusions?",
    "List important dates, names, or figures mentioned",
    "Explain the document in simple terms",
]
