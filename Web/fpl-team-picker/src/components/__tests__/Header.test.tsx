/**
 * @jest-environment jsdom
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
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

// Mock window.location.reload
const mockReload = jest.fn();

// Mock localStorage
const mockLocalStorage = {
  removeItem: jest.fn(),
};

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

  // Mock window globals in a Jest-friendly way
  const mockWindowLocation = {
    reload: jest.fn(),
  };

  const mockLocalStorage = {
    removeItem: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
    // Reset environment variable
    process.env.NEXT_PUBLIC_FPL_ASSISTANT_ENABLED = 'false';
    
    // Setup window mocks safely
    global.window = Object.create(window);
    Object.defineProperty(global.window, 'location', {
      value: mockWindowLocation,
      writable: true,
    });
    
    Object.defineProperty(global.window, 'localStorage', {
      value: mockLocalStorage,
      writable: true,
    });
  });

  describe('Rendering', () => {
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

  describe('Assistant Link', () => {
    test('shows assistant link when assistant is enabled', () => {
      process.env.NEXT_PUBLIC_FPL_ASSISTANT_ENABLED = 'true';
      
      renderHeader();
      
      const assistantLink = screen.getByText('Assistant (beta)');
      expect(assistantLink).toBeInTheDocument();
      expect(assistantLink.closest('a')).toHaveAttribute('href', '/assistant');
    });

    test('hides assistant link when assistant is disabled', () => {
      process.env.NEXT_PUBLIC_FPL_ASSISTANT_ENABLED = 'false';
      
      renderHeader();
      
      expect(screen.queryByText('Assistant (beta)')).not.toBeInTheDocument();
    });

    test('hides assistant link when environment variable is not set', () => {
      delete process.env.NEXT_PUBLIC_FPL_ASSISTANT_ENABLED;
      
      renderHeader();
      
      expect(screen.queryByText('Assistant (beta)')).not.toBeInTheDocument();
    });
  });

  describe('User ID Popover', () => {
    test('opens popover when user ID button is clicked', async () => {
      const user = userEvent.setup();
      renderHeader();
      
      const userIdButton = screen.getByText('12345');
      await user.click(userIdButton);
      
      await waitFor(() => {
        expect(screen.getByText('Reset')).toBeInTheDocument();
      });
    });

    test('reset button clears localStorage and reloads page', async () => {
      const user = userEvent.setup();
      renderHeader();
      
      // Open popover
      const userIdButton = screen.getByText('12345');
      await user.click(userIdButton);
      
      // Click reset button
      await waitFor(() => {
        expect(screen.getByText('Reset')).toBeInTheDocument();
      });
      
      const resetButton = screen.getByText('Reset');
      await user.click(resetButton);
      
      expect(mockLocalStorage.removeItem).toHaveBeenCalledWith('pl_profile');
      expect(mockWindowLocation.reload).toHaveBeenCalled();
    });

    test('reset button includes arrow icon', async () => {
      const user = userEvent.setup();
      renderHeader();
      
      const userIdButton = screen.getByText('12345');
      await user.click(userIdButton);
      
      await waitFor(() => {
        expect(screen.getByTestId('arrow-path-icon')).toBeInTheDocument();
      });
    });
  });

  describe('Styling', () => {
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
  });

  describe('Accessibility', () => {
    test('header has proper semantic role', () => {
      renderHeader();
      
      expect(screen.getByRole('banner')).toBeInTheDocument();
    });

    test('user ID button has proper focus management', async () => {
      const user = userEvent.setup();
      renderHeader();
      
      const userIdButton = screen.getByText('12345');
      await user.tab();
      
      // Button should be focusable
      expect(userIdButton).toHaveFocus();
    });

    test('reset button is accessible via keyboard', async () => {
      const user = userEvent.setup();
      renderHeader();
      
      const userIdButton = screen.getByText('12345');
      await user.click(userIdButton);
      
      await waitFor(() => {
        const resetButton = screen.getByText('Reset');
        expect(resetButton).toBeInTheDocument();
      });
      
      const resetButton = screen.getByText('Reset');
      await user.click(resetButton);
      
      expect(mockLocalStorage.removeItem).toHaveBeenCalled();
    });
  });
});
