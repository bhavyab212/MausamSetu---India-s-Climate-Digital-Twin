import type { ZodTypeAny, z } from "zod"

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8011/api/v1"

export class ApiError extends Error {
  status: number
  detail: unknown

  constructor(message: string, status: number, detail: unknown) {
    super(message)
    this.name = "ApiError"
    this.status = status
    this.detail = detail
  }
}

type QueryValue = string | number | boolean | null | undefined
export type QueryParams = Record<string, QueryValue | readonly QueryValue[]>

function buildUrl(path: string, params?: QueryParams): string {
  const url = new URL(path.startsWith("http") ? path : `${API_BASE_URL}${path}`)
  if (!params) return url.toString()
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null) continue
    if (Array.isArray(value)) {
      for (const item of value) {
        if (item === undefined || item === null) continue
        url.searchParams.append(key, String(item))
      }
    } else {
      url.searchParams.set(key, String(value))
    }
  }
  return url.toString()
}

interface RequestOptions<Schema extends ZodTypeAny> {
  path: string
  method?: "GET" | "POST"
  params?: QueryParams
  body?: unknown
  schema: Schema
  init?: RequestInit
}

async function apiRequest<Schema extends ZodTypeAny>({
  path,
  method = "GET",
  params,
  body,
  schema,
  init,
}: RequestOptions<Schema>): Promise<z.infer<Schema>> {
  const url = buildUrl(path, params)
  const response = await fetch(url, {
    method,
    headers: {
      "content-type": "application/json",
      accept: "application/json",
      ...(init?.headers ?? {}),
    },
    body: body !== undefined ? JSON.stringify(body) : undefined,
    ...init,
  })

  const contentType = response.headers.get("content-type") ?? ""
  const isJson = contentType.includes("application/json")
  const payload = isJson ? await response.json() : await response.text()

  if (!response.ok) {
    const message =
      typeof payload === "object" && payload && "detail" in payload
        ? String((payload as { detail: unknown }).detail)
        : `Request failed with status ${response.status}`
    throw new ApiError(message, response.status, payload)
  }

  return schema.parse(payload)
}

export const apiGet = <Schema extends ZodTypeAny>(
  path: string,
  schema: Schema,
  params?: QueryParams,
): Promise<z.infer<Schema>> => apiRequest({ path, schema, params, method: "GET" })

export const apiPost = <Schema extends ZodTypeAny>(
  path: string,
  schema: Schema,
  body: unknown,
): Promise<z.infer<Schema>> => apiRequest({ path, schema, method: "POST", body })

export function apiFetcher<Schema extends ZodTypeAny>(schema: Schema) {
  return async ([path, params]: readonly [string, QueryParams?]) =>
    apiGet(path, schema, params)
}
