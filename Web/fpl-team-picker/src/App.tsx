import './App.scss'
import { useLocalStorage } from "@uidotdev/usehooks";
import AuthGuard from "./components/AuthGuard.tsx";
import {FplApi} from "./helpers/fpl-api.ts";

function App() {
    const [plProfile, savePlProfile] = useLocalStorage<string | null>("pl_profile", null);

    if (!plProfile) {
        return <AuthGuard onDone={(cookie) => {
            debugger;
            savePlProfile(cookie)
        }}/>
    }
    new FplApi().me.getMe().then(res => console.log(res.data));

    return (
        <>
            <h1>
                Hello world!
            </h1>
        </>
    )
}

export default App
