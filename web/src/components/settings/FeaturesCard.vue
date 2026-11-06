<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { systemApi } from '@/api/system.ts'
import { useToastStore } from '@/stores/toast.ts'
import Button from '@/components/ui/Button.vue'
import ConfigSelect from '@/components/config/ConfigSelect.vue'
import ConfigSwitch from '@/components/config/ConfigSwitch.vue'
import ConfigInput from '@/components/config/ConfigInput.vue'
import { ShieldCheck, Save, RefreshCw } from 'lucide-vue-next'

const toast = useToastStore()

const loading = ref(true)
const saving = ref(false)

// ATG 抗截断
const atgMode = ref('off')
const atgModeGemini = ref('inherit')
const atgModeOpenai = ref('inherit')
const atgEmptyRetries = ref(2)

// OpenAI 协议
const openaiEnabled = ref(true)

const atgModes = [
  { value: 'off', label: '关闭 (off)', description: '不注入合成通道' },
  { value: 'prefer', label: '优先 (prefer)', description: '文本模型默认注入,失败自动回退' },
  { value: 'required', label: '强制 (required)', description: '必须走合成通道,失败则报错' },
]

const inheritModes = [
  { value: 'inherit', label: '继承全局 (inherit)', description: '跟随上方全局模式' },
  { value: 'off', label: '关闭 (off)' },
  { value: 'prefer', label: '优先 (prefer)' },
  { value: 'required', label: '强制 (required)' },
]

function applyFeatures(features: Record<string, unknown>) {
  const atg = (features.atg ?? {}) as Record<string, unknown>
  atgMode.value = String(atg.mode ?? 'off')
  const byFormat = (atg.mode_by_format ?? {}) as Record<string, string>
  atgModeGemini.value = byFormat.gemini ?? 'inherit'
  atgModeOpenai.value = byFormat.openai ?? 'inherit'
  atgEmptyRetries.value = Number(atg.empty_retries ?? 2)
  const formats = (features.formats ?? {}) as Record<string, unknown>
  openaiEnabled.value = Boolean(formats.openai ?? true)
}

async function loadFeatures() {
  loading.value = true
  try {
    const res = await systemApi.getFeatures()
    applyFeatures(res.features as unknown as Record<string, unknown>)
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : '加载功能开关失败'
    toast.error(msg)
  } finally {
    loading.value = false
  }
}

async function handleSave() {
  saving.value = true
  try {
    const res = await systemApi.updateFeatures({
      atg: {
        mode: atgMode.value,
        mode_by_format: {
          gemini: atgModeGemini.value,
          openai: atgModeOpenai.value,
        },
        empty_retries: atgEmptyRetries.value,
      },
      formats: {
        openai: openaiEnabled.value,
      },
    })
    applyFeatures(res.features as unknown as Record<string, unknown>)
    toast.success('功能开关已保存并热生效')
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : '保存功能开关失败'
    toast.error(msg)
  } finally {
    saving.value = false
  }
}

onMounted(loadFeatures)
</script>

<template>
  <div class="bg-white border border-gray-200/80 rounded-2xl p-6 shadow-xs space-y-5">
    <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-gray-100 pb-4">
      <div class="flex items-center gap-2.5">
        <div class="p-2 rounded-xl bg-amber-50 text-amber-600 border border-amber-100">
          <ShieldCheck class="w-5 h-5" />
        </div>
        <div>
          <h3 class="font-semibold text-gray-900 text-sm">功能开关</h3>
          <p class="text-xs text-gray-500 mt-0.5">ATG 抗截断 / OpenAI 协议 / 账号双保活,保存后立即热生效</p>
        </div>
      </div>
      <div class="flex items-center gap-2">
        <Button variant="secondary" size="sm" :loading="loading" @click="loadFeatures">
          <RefreshCw class="w-3.5 h-3.5" />
          <span>刷新</span>
        </Button>
        <Button variant="primary" size="sm" :loading="saving" @click="handleSave">
          <Save class="w-3.5 h-3.5" />
          <span>保存功能开关</span>
        </Button>
      </div>
    </div>

    <!-- ATG 抗截断 -->
    <div class="p-4 bg-gray-50/60 border border-gray-200/70 rounded-2xl space-y-4">
      <div class="flex items-center gap-1.5 text-xs font-semibold text-gray-700">
        <ShieldCheck class="w-3.5 h-3.5 text-brand-600" />
        <span>ATG 抗截断(合成工具函数通道,防长文截断)</span>
      </div>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <ConfigSelect
          :model-value="atgMode"
          label="全局模式"
          :options="atgModes"
          @update:model-value="atgMode = String($event || 'off')"
        />
        <ConfigInput
          :model-value="atgEmptyRetries"
          label="空响应重发次数 (0-5)"
          placeholder="2"
          @update:model-value="atgEmptyRetries = Number($event || 0)"
        />
        <ConfigSelect
          :model-value="atgModeGemini"
          label="Gemini 协议面覆盖"
          :options="inheritModes"
          @update:model-value="atgModeGemini = String($event || 'inherit')"
        />
        <ConfigSelect
          :model-value="atgModeOpenai"
          label="OpenAI 协议面覆盖"
          :options="inheritModes"
          @update:model-value="atgModeOpenai = String($event || 'inherit')"
        />
      </div>
    </div>

    <!-- OpenAI 协议 -->
    <div class="p-4 bg-gray-50/60 border border-gray-200/70 rounded-2xl space-y-2">
      <ConfigSwitch
        :model-value="openaiEnabled"
        label="OpenAI 兼容协议 (/v1/chat/completions)"
        description="关闭后 /v1/* 接口返回 403 feature_disabled"
        @update:model-value="openaiEnabled = $event"
      />
    </div>
  </div>
</template>
