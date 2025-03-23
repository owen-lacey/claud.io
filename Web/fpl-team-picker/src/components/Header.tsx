import { memo, useContext } from "react";
import { DataContext } from "../App.tsx";

const Header = memo(function Header() {
    const allData = useContext(DataContext);

    if (!allData?.myDetails) {
        return <></>
    }
    const { myDetails } = allData;

    return <header className="bg-white border border-gray-300 shadow-lg rounded-lg py-4 px-6 flex justify-between items-center">
        <h1 className="text-2xl font-semibold">Welcome, <span>{myDetails.firstName}</span>!</h1>
        <p className="text-sm">User ID: <span className="font-mono bg-gray-100 px-2 py-1 rounded-md">{myDetails.id}</span></p>
    </header>

});

export default Header;
