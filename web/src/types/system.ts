export interface SystemConfig {
  port: number
  signer_mode?: string
  signer_ready?: boolean
  signer_running?: boolean
  proxy_configured: boolean
  auth_enabled: boolean
  yaml_content: string
  log_level?: string
  dump_requests?: boolean
  debug_env_active?: boolean
  dump_to_file?: boolean
  dump_dir?: string
}
export interface UpdateLoggingConfigRequest {
  level?: string
  dump_requests?: boolean
  dump_to_file?: boolean
  dump_dir?: string
}
export interface UpdateLoggingConfigResponse {
  ok: boolean
  log_level: string
  dump_requests: boolean
  debug_env_active: boolean
  dump_to_file: boolean
  dump_dir: string
}

export interface HealthCheckResponse {
  status: string
}
export interface ApiKeyItem {
  name: string
  key: string
  created_at?: string | null
}

export interface CreateApiKeyRequest {
  name?: string
  key?: string
}

export interface UpdateApiKeyRequest {
  name: string
}

export interface AtgFeatureConfig {
  mode: string
  mode_by_format?: Record<string, string>
  tool_prefix?: string
  control_position?: string
  strict?: boolean
  empty_retries?: number
  recovery?: string
  side_buffer_bytes?: number
  client_synthetic_guard?: boolean
  max_response_bytes?: number
  text_model_pattern?: string
  non_text_model_pattern?: string
}

export interface FeaturesConfig {
  formats: { openai: boolean }
  atg: AtgFeatureConfig
}

export interface FeaturesConfigResponse {
  features: FeaturesConfig
}
