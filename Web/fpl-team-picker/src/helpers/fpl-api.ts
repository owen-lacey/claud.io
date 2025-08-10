import { AxiosError } from "axios";
import { Api, ApiConfig } from "./api";

export class FplApi extends Api<unknown> {
  constructor(providedToken?: string) {
    let token: string | undefined = providedToken;

    if (typeof window !== 'undefined') {
      try {
        token = token ?? localStorage.getItem('pl_profile')?.replace(/"/g, "");
      } catch {
        // ignore
      }
    }

    const config: ApiConfig = {
      baseURL: process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:5079',
      headers: {
        ...(token ? { pl_profile: token } : {}),
      }
    };

    super(config);

    this.instance.interceptors.response.use((res) => res, (error: AxiosError) => {
      const status = error.response?.status;
      if (status === 401 && typeof window !== 'undefined') {
        try { localStorage.removeItem('pl_profile'); } catch {}
      }
      return Promise.reject(error);
    });
  }
}