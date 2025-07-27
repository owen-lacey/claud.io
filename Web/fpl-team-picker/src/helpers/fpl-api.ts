import { AxiosError } from "axios";
import { Api, ApiConfig } from "./api";

export class FplApi extends Api<unknown> {
  constructor() {
    const cookieVal = localStorage.getItem('pl_profile')?.replace(/"/g, "");
    const config: ApiConfig = {
      baseURL: process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:5079',
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