import React, { forwardRef } from 'react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger' | 'outline';
  size?: 'sm' | 'md' | 'lg';
  icon?: React.ReactNode;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'secondary', size = 'md', icon, children, disabled, ...props }, ref) => {
    const baseStyles =
      'inline-flex items-center justify-center font-medium transition-colors duration-150 ease-out focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-1 disabled:opacity-40 disabled:cursor-not-allowed select-none rounded-[8px] min-h-[40px]';

    const sizeStyles = {
      sm: 'h-9 min-h-[36px] px-3 text-xs gap-1.5',
      md: 'h-10 min-h-[40px] px-4 text-sm gap-2',
      lg: 'h-11 min-h-[44px] px-5 text-base gap-2.5 font-medium',
    };

    const variantStyles = {
      primary:
        'bg-primary text-white hover:bg-primary-hover shadow-sm border border-transparent',
      secondary:
        'bg-surface hover:bg-elevated text-txt-primary border border-borderSubtle shadow-card hover:border-borderStrong',
      ghost:
        'bg-transparent hover:bg-elevated text-txt-secondary hover:text-txt-primary border border-transparent',
      danger:
        'bg-spoof-bg text-spoof border border-spoof/30 hover:bg-spoof/20 shadow-sm',
      outline:
        'bg-transparent text-primary border border-primary/30 hover:bg-primary/5 hover:border-primary',
    };

    return (
      <button
        ref={ref}
        disabled={disabled}
        className={twMerge(clsx(baseStyles, sizeStyles[size], variantStyles[variant], className))}
        {...props}
      >
        {icon && <span className="flex-shrink-0">{icon}</span>}
        {children}
      </button>
    );
  }
);

Button.displayName = 'Button';
