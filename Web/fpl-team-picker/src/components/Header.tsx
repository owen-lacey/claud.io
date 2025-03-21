import {FplApi} from "../helpers/fpl-api.ts";
import {memo, useEffect, useState} from "react";
import {User} from "../helpers/api.ts";

const Header = memo(function Header() {
    const [me, setMe] = useState<User | null>(null);

    const loadMyDetails = async () => {
        const result = await new FplApi().myDetails.myDetailsList();
        setMe(result.data);
    }

    useEffect(() => {
        loadMyDetails();
    }, []);

    if (!me) {
        return <></>
    }

    return <header className="bg-white border border-gray-300 shadow-lg rounded-lg py-4 px-6 flex justify-between items-center">
    <h1 className="text-2xl font-semibold">Welcome, <span>{me.firstName}</span>!</h1>
    <p className="text-sm">User ID: <span className="font-mono bg-gray-100 px-2 py-1 rounded-md">{me.id}</span></p>
</header>

});

export default Header;
