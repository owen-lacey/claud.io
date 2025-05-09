/* eslint-disable */
/* tslint:disable */
// @ts-nocheck
/*
 * ---------------------------------------------------------------
 * ## THIS FILE WAS GENERATED VIA SWAGGER-TYPESCRIPT-API        ##
 * ##                                                           ##
 * ## AUTHOR: acacode                                           ##
 * ## SOURCE: https://github.com/acacode/swagger-typescript-api ##
 * ---------------------------------------------------------------
 */

export interface League {
  /** @format int32 */
  id?: number;
  name?: string | null;
  /** @format int32 */
  currentPosition?: number;
  participants?: LeagueParticipant[] | null;
  /** @format int32 */
  numberOfPlayers?: number;
}

export interface LeagueParticipant {
  /** @format int32 */
  userId?: number;
  playerName?: string | null;
  teamName?: string | null;
  /** @format int32 */
  position?: number;
  /** @format int32 */
  total?: number;
}

export interface Manager {
  /** @format int32 */
  id?: number;
  /** @format int32 */
  cost?: number;
  firstName?: string | null;
  secondName?: string | null;
  /** @format double */
  xpNext?: number;
  /** @format double */
  xpThis?: number;
  /** @format int32 */
  team?: number;
  name?: string | null;
}

export interface MyTeam {
  /** @format int32 */
  freeTransfers?: number;
  /** @format int32 */
  bank?: number;
  selectedSquad?: SelectedSquad;
  /** @format int32 */
  budget?: number;
}

export interface Player {
  /** @format int32 */
  id?: number;
  position?: Position;
  /** @format int32 */
  cost?: number;
  /** @format int32 */
  chanceOfPlayingNextRound?: number | null;
  firstName?: string | null;
  secondName?: string | null;
  /** @format double */
  xpNext?: number;
  /** @format double */
  xpThis?: number;
  /** @format int32 */
  team?: number;
  /** @format int32 */
  seasonPoints?: number;
  name?: string | null;
  isAvailable?: boolean;
}

/** @format int32 */
export enum Position {
  Value1 = 1,
  Value2 = 2,
  Value3 = 3,
  Value4 = 4,
  Value5 = 5,
}

export interface SelectedPlayer {
  isViceCaptain?: boolean;
  isCaptain?: boolean;
  player?: Player;
  /** @format int32 */
  sellingPrice?: number;
}

export interface SelectedSquad {
  startingXi?: SelectedPlayer[] | null;
  bench?: SelectedPlayer[] | null;
  /** @format int32 */
  squadCost?: number;
  /** @format double */
  predictedPoints?: number;
  /** @format double */
  benchBoostPredictedPoints?: number;
}

export interface SelectedTeam {
  startingXi?: SelectedPlayer[] | null;
  /** @format int32 */
  squadCost?: number;
  /** @format double */
  score?: number;
}

export interface Team {
  shortName?: string | null;
  name?: string | null;
  /** @format int32 */
  code?: number;
  /** @format int32 */
  id?: number;
}

export interface Transfers {
  playersOut?: SelectedPlayer[] | null;
  playersIn?: SelectedPlayer[] | null;
  startingXi?: SelectedPlayer[] | null;
  bench?: SelectedPlayer[] | null;
  /** @format int32 */
  freeTransfers?: number;
  /** @format int32 */
  bank?: number;
  /** @format int32 */
  squadCost?: number;
  /** @format int32 */
  budget?: number;
  /** @format double */
  predictedPoints?: number;
}

export interface User {
  firstName?: string | null;
  lastName?: string | null;
  /** @format int32 */
  id?: number;
}

import type { AxiosInstance, AxiosRequestConfig, AxiosResponse, HeadersDefaults, ResponseType } from "axios";
import axios from "axios";

export type QueryParamsType = Record<string | number, any>;

export interface FullRequestParams extends Omit<AxiosRequestConfig, "data" | "params" | "url" | "responseType"> {
  /** set parameter to `true` for call `securityWorker` for this request */
  secure?: boolean;
  /** request path */
  path: string;
  /** content type of request body */
  type?: ContentType;
  /** query params */
  query?: QueryParamsType;
  /** format of response (i.e. response.json() -> format: "json") */
  format?: ResponseType;
  /** request body */
  body?: unknown;
}

export type RequestParams = Omit<FullRequestParams, "body" | "method" | "query" | "path">;

export interface ApiConfig<SecurityDataType = unknown> extends Omit<AxiosRequestConfig, "data" | "cancelToken"> {
  securityWorker?: (
    securityData: SecurityDataType | null,
  ) => Promise<AxiosRequestConfig | void> | AxiosRequestConfig | void;
  secure?: boolean;
  format?: ResponseType;
}

export enum ContentType {
  Json = "application/json",
  FormData = "multipart/form-data",
  UrlEncoded = "application/x-www-form-urlencoded",
  Text = "text/plain",
}

export class HttpClient<SecurityDataType = unknown> {
  public instance: AxiosInstance;
  private securityData: SecurityDataType | null = null;
  private securityWorker?: ApiConfig<SecurityDataType>["securityWorker"];
  private secure?: boolean;
  private format?: ResponseType;

  constructor({ securityWorker, secure, format, ...axiosConfig }: ApiConfig<SecurityDataType> = {}) {
    this.instance = axios.create({ ...axiosConfig, baseURL: axiosConfig.baseURL || "" });
    this.secure = secure;
    this.format = format;
    this.securityWorker = securityWorker;
  }

  public setSecurityData = (data: SecurityDataType | null) => {
    this.securityData = data;
  };

  protected mergeRequestParams(params1: AxiosRequestConfig, params2?: AxiosRequestConfig): AxiosRequestConfig {
    const method = params1.method || (params2 && params2.method);

    return {
      ...this.instance.defaults,
      ...params1,
      ...(params2 || {}),
      headers: {
        ...((method && this.instance.defaults.headers[method.toLowerCase() as keyof HeadersDefaults]) || {}),
        ...(params1.headers || {}),
        ...((params2 && params2.headers) || {}),
      },
    };
  }

  protected stringifyFormItem(formItem: unknown) {
    if (typeof formItem === "object" && formItem !== null) {
      return JSON.stringify(formItem);
    } else {
      return `${formItem}`;
    }
  }

  protected createFormData(input: Record<string, unknown>): FormData {
    if (input instanceof FormData) {
      return input;
    }
    return Object.keys(input || {}).reduce((formData, key) => {
      const property = input[key];
      const propertyContent: any[] = property instanceof Array ? property : [property];

      for (const formItem of propertyContent) {
        const isFileType = formItem instanceof Blob || formItem instanceof File;
        formData.append(key, isFileType ? formItem : this.stringifyFormItem(formItem));
      }

      return formData;
    }, new FormData());
  }

  public request = async <T = any, _E = any>({
    secure,
    path,
    type,
    query,
    format,
    body,
    ...params
  }: FullRequestParams): Promise<AxiosResponse<T>> => {
    const secureParams =
      ((typeof secure === "boolean" ? secure : this.secure) &&
        this.securityWorker &&
        (await this.securityWorker(this.securityData))) ||
      {};
    const requestParams = this.mergeRequestParams(params, secureParams);
    const responseFormat = format || this.format || undefined;

    if (type === ContentType.FormData && body && body !== null && typeof body === "object") {
      body = this.createFormData(body as Record<string, unknown>);
    }

    if (type === ContentType.Text && body && body !== null && typeof body !== "string") {
      body = JSON.stringify(body);
    }

    return this.instance.request({
      ...requestParams,
      headers: {
        ...(requestParams.headers || {}),
        ...(type ? { "Content-Type": type } : {}),
      },
      params: query,
      responseType: responseFormat,
      data: body,
      url: path,
    });
  };
}

/**
 * @title FplTeamPicker.Api
 * @version 1.0
 */
export class Api<SecurityDataType extends unknown> extends HttpClient<SecurityDataType> {
  transfers = {
    /**
     * No description
     *
     * @tags FplTeamPicker.Api
     * @name TransfersCreate
     * @request POST:/transfers
     */
    transfersCreate: (params: RequestParams = {}) =>
      this.request<Transfers, any>({
        path: `/transfers`,
        method: "POST",
        format: "json",
        ...params,
      }),
  };
  wildcard = {
    /**
     * No description
     *
     * @tags FplTeamPicker.Api
     * @name WildcardCreate
     * @request POST:/wildcard
     */
    wildcardCreate: (params: RequestParams = {}) =>
      this.request<MyTeam, any>({
        path: `/wildcard`,
        method: "POST",
        format: "json",
        ...params,
      }),
  };
  tots = {
    /**
     * No description
     *
     * @tags FplTeamPicker.Api
     * @name TotsCreate
     * @request POST:/tots
     */
    totsCreate: (params: RequestParams = {}) =>
      this.request<SelectedTeam, any>({
        path: `/tots`,
        method: "POST",
        format: "json",
        ...params,
      }),
  };
  myDetails = {
    /**
     * No description
     *
     * @tags FplTeamPicker.Api
     * @name MyDetailsList
     * @request GET:/my-details
     */
    myDetailsList: (params: RequestParams = {}) =>
      this.request<User, any>({
        path: `/my-details`,
        method: "GET",
        format: "json",
        ...params,
      }),
  };
  myTeam = {
    /**
     * No description
     *
     * @tags FplTeamPicker.Api
     * @name MyTeamList
     * @request GET:/my-team
     */
    myTeamList: (params: RequestParams = {}) =>
      this.request<MyTeam, any>({
        path: `/my-team`,
        method: "GET",
        format: "json",
        ...params,
      }),
  };
  myLeagues = {
    /**
     * No description
     *
     * @tags FplTeamPicker.Api
     * @name MyLeaguesList
     * @request GET:/my-leagues
     */
    myLeaguesList: (params: RequestParams = {}) =>
      this.request<League[], any>({
        path: `/my-leagues`,
        method: "GET",
        format: "json",
        ...params,
      }),
  };
  teams = {
    /**
     * No description
     *
     * @tags FplTeamPicker.Api
     * @name TeamsList
     * @request GET:/teams
     */
    teamsList: (params: RequestParams = {}) =>
      this.request<Team[], any>({
        path: `/teams`,
        method: "GET",
        format: "json",
        ...params,
      }),
  };
  players = {
    /**
     * No description
     *
     * @tags FplTeamPicker.Api
     * @name PlayersList
     * @request GET:/players
     */
    playersList: (params: RequestParams = {}) =>
      this.request<Player[], any>({
        path: `/players`,
        method: "GET",
        format: "json",
        ...params,
      }),
  };
  managers = {
    /**
     * No description
     *
     * @tags FplTeamPicker.Api
     * @name ManagersList
     * @request GET:/managers
     */
    managersList: (params: RequestParams = {}) =>
      this.request<Manager[], any>({
        path: `/managers`,
        method: "GET",
        format: "json",
        ...params,
      }),
  };
  users = {
    /**
     * No description
     *
     * @tags FplTeamPicker.Api
     * @name CurrentTeamList
     * @request GET:/users/{userId}/current-team
     */
    currentTeamList: (userId: number, params: RequestParams = {}) =>
      this.request<SelectedSquad, any>({
        path: `/users/${userId}/current-team`,
        method: "GET",
        format: "json",
        ...params,
      }),
  };
}
