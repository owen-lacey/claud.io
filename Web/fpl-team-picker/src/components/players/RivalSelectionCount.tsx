import { PlayerSelection } from "../../models/player-selection";

function RivalSelectionCount({selections, totalRivals}: {selections: PlayerSelection[], totalRivals: number}) {

    const percentSelected = ((1 - selections.length / totalRivals) * 100).toFixed(0);

    return <div className="selection-counter">
        <div className="bg-linear-to-r from-cyan-500 to-blue-500" style={{right: `${percentSelected}%`}}></div>
    </div>
}

export default RivalSelectionCount;