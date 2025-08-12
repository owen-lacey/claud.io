/**
 * @jest-environment jsdom
 */

import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import Header from '../Header';
import { DataContext } from '@/lib/contexts';
import { AllData } from '@/models/all-data';
import { ApiResult } from '@/models/api-result';

// Mock Next.js Link component
jest.mock('next/link', () => {
  const Link = ({ children, href, ...props }: any) => (
    <a href={href} {...props}>{children}</a>
  );
  Link.displayName = 'Link';
  return Link;
});

// Mock Heroicons
jest.mock('@heroicons/react/24/solid', () => ({
  ArrowPathRoundedSquareIcon: ({ className }: { className?: string }) => (
    <div data-testid="arrow-path-icon" className={className}>ArrowPathIcon</div>
  ),
}));

describe('Header Component', () => {
  const mockContextValue: AllData = {
    myDetails: new ApiResult(true, {
      id: 12345,
      firstName: 'John',
      lastName: 'Doe'
    }),
    players: new ApiResult(true, []),
    teams: new ApiResult(true, []),
    leagues: new ApiResult(true, []),
    myTeam: new ApiResult(true, { freeTransfers: 1, bank: 0, budget: 1000 })
  };

  const renderHeader = (contextValue: AllData | null = mockContextValue) => {
    return render(
      <DataContext.Provider value={contextValue}>
        <Header />
      </DataContext.Provider>
    );
  };

  beforeEach(() => {
    // Reset environment variable
    process.env.NEXT_PUBLIC_FPL_ASSISTANT_ENABLED = 'false';
  });

  describe('Basic Rendering', () => {
    test('renders welcome message with user first name', () => {
      renderHeader();
      
      expect(screen.getByText(/Welcome,/)).toBeInTheDocument();
      expect(screen.getByText('John')).toBeInTheDocument();
    });

    test('displays user ID in popover button', () => {
      renderHeader();
      
      expect(screen.getByText('12345')).toBeInTheDocument();
      expect(screen.getByText(/User ID:/)).toBeInTheDocument();
    });

    test('shows loading state when myDetails is not available', () => {
      const loadingContextValue: AllData = {
        ...mockContextValue,
        myDetails: new ApiResult(true, null as any)
      };
      
      renderHeader(loadingContextValue);
      
      // Should render LoadingCard instead of header content
      expect(screen.queryByText(/Welcome,/)).not.toBeInTheDocument();
    });
  });

  describe('Styling and Accessibility', () => {
    test('applies correct CSS classes to header element', () => {
      renderHeader();
      
      const headerElement = screen.getByRole('banner');
      expect(headerElement).toHaveClass(
        'bg-card',
        'border',
        'border-border',
        'shadow-lg',
        'rounded-lg',
        'py-4',
        'px-6'
      );
    });

    test('welcome text has correct styling', () => {
      renderHeader();
      
      const welcomeHeader = screen.getByRole('heading', { level: 1 });
      expect(welcomeHeader).toHaveClass('text-2xl', 'font-semibold', 'text-foreground');
    });

    test('header has proper semantic role', () => {
      renderHeader();
      
      expect(screen.getByRole('banner')).toBeInTheDocument();
    });
  });
});
