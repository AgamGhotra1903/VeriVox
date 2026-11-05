/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        chassis: "var(--bg-chassis)",
        surface: "var(--bg-surface)",
        elevated: "var(--bg-elevated)",
        inset: "var(--bg-inset)",
        borderSubtle: "var(--border-subtle)",
        borderStrong: "var(--border-strong)",
        primary: {
          DEFAULT: "var(--accent-primary)",
          hover: "var(--accent-hover)",
          subtle: "var(--accent-subtle)",
          ring: "var(--accent-ring)",
        },
        genuine: {
          DEFAULT: "var(--status-genuine)",
          bg: "var(--status-genuine-bg)",
          border: "var(--status-genuine-border)",
        },
        spoof: {
          DEFAULT: "var(--status-spoof)",
          bg: "var(--status-spoof-bg)",
          border: "var(--status-spoof-border)",
        },
        uncertain: {
          DEFAULT: "var(--status-uncertain)",
          bg: "var(--status-uncertain-bg)",
          border: "var(--status-uncertain-border)",
        },
        txt: {
          primary: "var(--text-primary)",
          secondary: "var(--text-secondary)",
          muted: "var(--text-muted)",
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', '"Segoe UI"', 'Roboto', 'sans-serif'],
        display: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', '"Segoe UI"', 'Roboto', 'sans-serif'],
        body: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', '"Segoe UI"', 'Roboto', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'monospace'],
      },
      boxShadow: {
        'card': 'var(--shadow-card)',
        'subtle': 'var(--shadow-sm)',
      },
      borderRadius: {
        'card': 'var(--radius-lg)',
        'btn': 'var(--radius-md)',
      }
    },
  },
  plugins: [],
}
