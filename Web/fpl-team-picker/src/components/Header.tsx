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

    return <header className="bg-white">
        <nav className="flex">
            <div className="text-xl font-semibold">Welcome, {me.firstName}</div>
            <div className="text-sm blue-500 ml-4 italic">#{me.id}</div>
        </nav>
    </header>
});

export default Header;
