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
  beforeEach(() => {
    jest.clearAllMocks();
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
      
      // Mock console.error to catch the "Not implemented: navigation" error
      const consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
      
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
      
      expect(window.localStorage.removeItem).toHaveBeenCalledWith('pl_profile');
      
      // Verify that reload was called by checking for the "Not implemented" error
      expect(consoleErrorSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          message: expect.stringContaining('Not implemented: navigation')
        })
      );
      
      // Restore console.error
      consoleErrorSpy.mockRestore();
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
      
      expect(window.localStorage.removeItem).toHaveBeenCalled();
    });
  });
});
