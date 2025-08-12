/**
 * @jest-environment jsdom
 */

import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import { LoadingCard } from '../utils/Loading';

describe('LoadingCard Component', () => {
  test('renders loading spinner and text', () => {
    render(<LoadingCard />);
    
    expect(screen.getByText('Loading...')).toBeInTheDocument();
    expect(screen.getByRole('status')).toBeInTheDocument();
  });

  test('has proper accessibility attributes', () => {
    render(<LoadingCard />);
    
    const statusElement = screen.getByRole('status');
    expect(statusElement).toHaveAttribute('aria-live', 'polite');
  });

  test('applies correct CSS classes', () => {
    render(<LoadingCard />);
    
    const statusElement = screen.getByRole('status');
    const cardElement = statusElement.parentElement;
    expect(cardElement).toHaveClass('bg-card', 'border', 'border-border');
  });
});
