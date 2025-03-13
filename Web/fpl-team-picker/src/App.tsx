import './App.scss'
import { useLocalStorage } from "@uidotdev/usehooks";
import AuthGuard from "./components/AuthGuard.tsx";
import Header from "./components/Header.tsx";
import {memo} from "react";

const App = memo(function App() {
    const [plProfile, savePlProfile] = useLocalStorage<string | null>("pl_profile", null);

    if (!plProfile) {
        return <AuthGuard onDone={(cookie) => savePlProfile(cookie)}/>
    }

    return (
        <div className="app-container">
            <div className="header">
                <Header/>
            </div>
            <div className="my-team"></div>
            <div className="players"></div>
            <div className="leagues"></div>
        </div>
    )
});

export default App
