import { useContext } from "react";
import { DataContext } from "../App";

function Leagues() {
    const allData = useContext(DataContext);

    if (!allData) {
        return <></>;
    }
    const { leagues } = allData;

    return <div>
        <div>
            {leagues.map((league, i) => {
                return <div className="flex gap-1" key={`${i}-xi`}>
                    <div>{league!.name}</div>
                    <div>{league.currentPosition}</div>
                    <div>{league.numberOfPlayers}</div>
                </div>
            })}
        </div>
    </div>
}

export default Leagues;