import { AxiosError } from "axios";
import { Api, ApiConfig } from "./api.ts";

export class FplApi extends Api<unknown> {
  constructor() {
    const cookieVal = localStorage.getItem('pl_profile')?.replaceAll("\"", "");
    const config: ApiConfig = {
      baseURL: import.meta.env.VITE_API_BASE_URL,
      headers: {
        pl_profile: cookieVal
      }
    };

    super(config);

    this.instance.interceptors.response.use((res) => res, (error: AxiosError) => {
      if (error.status === 401) {
        localStorage.removeItem('pl_profile');
      }
    });
  }
}