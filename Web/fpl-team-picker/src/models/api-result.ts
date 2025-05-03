export class ApiResult<T> {
  constructor(success: boolean, result: (T | string)) {
    if (success) {
      this.output = result as T;
    } else {
      this.error = result as string;
    }
  }

  get isSuccess(): boolean { return !this.error; };
  error: string = null!;
  output: T | null = null;
}