import { render, screen } from '@testing-library/react';
import SquadPanel from '../SquadPanel';

// Mock the data service
jest.mock('@/lib/data-service', () => ({
  dataService: {
    fromToolSquad: jest.fn(),
    getWildcardRecommendation: jest.fn(),
  },
}));

describe('SquadPanel', () => {
  it('renders loading state initially', () => {
    render(<SquadPanel />);
    expect(screen.getByRole('button', { name: /refresh/i })).toBeInTheDocument();
  });

  it('renders with toolSquad prop', () => {
    const mockSquad = {
      startingXi: [],
      bench: [],
      squadCost: 950,
      predictedPoints: 65.2,
      benchBoostPredictedPoints: 68.1
    };
    
    render(<SquadPanel toolSquad={mockSquad} header="Test Squad" />);
    expect(screen.getByText('Test Squad')).toBeInTheDocument();
    expect(screen.getByText('AI-generated squad from chat tools')).toBeInTheDocument();
  });

  it('renders without toolSquad prop', () => {
    render(<SquadPanel />);
    expect(screen.getByText('AI Squad Recommendation')).toBeInTheDocument();
  });
});
