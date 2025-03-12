import {Api, ApiConfig} from "./api.ts";

export class FplApi extends Api<unknown> {
    constructor() {
        let cookieVal = localStorage.getItem('pl_profile')?.replaceAll("\"", "");
        const config: ApiConfig = {
            baseURL: import.meta.env.VITE_API_BASE_URL,
            headers: {
                pl_profile: cookieVal
            }
        };
        super(config);
    }
}