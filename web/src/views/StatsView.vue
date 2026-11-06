<script setup lang="ts">
import { computed, ref } from 'vue'
import {
  RefreshCw,
  ChevronDown,
  ChevronRight,
  BarChart3,
  Inbox,
} from 'lucide-vue-next'
import { useSystemStore } from '@/stores/system.ts'
import { usePolling } from '@/composables/usePolling.ts'
import type { ModelStatItem } from '@/types'
import { formatDate } from '@/utils/format.ts'

const systemStore = useSystemStore()
const expandedModels = ref<Set<string>>(new Set())

const loading = computed(() => !systemStore.stats)

const modelRows = computed(() => {
  const models = systemStore.stats?.models ?? {}
  return Object.entries(models).sort(
    (a, b) => (b[1].requests || 0) - (a[1].requests || 0),
  )
})

const totals = computed(() => systemStore.stats?.totals)

function fmt(n: number | undefined | null): string {
  return (n || 0).toLocaleString()
}

function pct(n: number | undefined, d: number | undefined): string {
  if (!d) return '0%'
  return `${((n || 0) / d * 100).toFixed(1)}%`
}

function toggleModel(name: string) {
  const next = new Set(expandedModels.value)
  if (next.has(name)) next.delete(name)
  else next.add(name)
  expandedModels.value = next
}

function hasAccountBreakdown(model: ModelStatItem): boolean {
  return Array.isArray(model.account_breakdown) && model.account_breakdown.length > 0
}

async function loadData() {
  await systemStore.fetchStats()
}

usePolling(loadData, 10000)
</script>

<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
      <div class="flex items-center gap-2.5">
        <div class="p-2 rounded-xl bg-blue-50 text-blue-600 border border-blue-100">
          <BarChart3 class="w-5 h-5" />
        </div>
        <div>
          <h2 class="font-semibold text-gray-900">用量统计</h2>
          <p class="text-xs text-gray-500 mt-0.5">模型调用量与 Token 消耗(累计,10 秒自动刷新)</p>
        </div>
      </div>
      <button
        type="button"
        class="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg border border-gray-200 text-gray-700 bg-white hover:bg-gray-50 transition-colors"
        @click="loadData"
      >
        <RefreshCw class="w-3.5 h-3.5" />
        立即刷新
      </button>
    </div>

    <!-- Totals tiles -->
    <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
      <div class="bg-white border border-gray-200/80 rounded-2xl p-4 shadow-xs">
        <div class="text-[11px] text-gray-500">总请求</div>
        <div class="text-xl font-bold text-gray-900 mt-1">{{ fmt(totals?.requests) }}</div>
        <div class="text-[11px] text-emerald-600 mt-0.5">成功 {{ fmt(totals?.success) }}</div>
      </div>
      <div class="bg-white border border-gray-200/80 rounded-2xl p-4 shadow-xs">
        <div class="text-[11px] text-gray-500">429 配额</div>
        <div class="text-xl font-bold mt-1" :class="(totals?.rate_limited || 0) > 0 ? 'text-amber-600' : 'text-gray-900'">
          {{ fmt(totals?.rate_limited) }}
        </div>
        <div class="text-[11px] text-gray-500 mt-0.5">占比 {{ pct(totals?.rate_limited, totals?.requests) }}</div>
      </div>
      <div class="bg-white border border-gray-200/80 rounded-2xl p-4 shadow-xs">
        <div class="text-[11px] text-gray-500">错误</div>
        <div class="text-xl font-bold mt-1" :class="(totals?.errors || 0) > 0 ? 'text-red-600' : 'text-gray-900'">
          {{ fmt(totals?.errors) }}
        </div>
        <div class="text-[11px] text-gray-500 mt-0.5">占比 {{ pct(totals?.errors, totals?.requests) }}</div>
      </div>
      <div class="bg-white border border-gray-200/80 rounded-2xl p-4 shadow-xs">
        <div class="text-[11px] text-gray-500">输入 Tokens</div>
        <div class="text-xl font-bold text-sky-600 mt-1">{{ fmt(totals?.prompt_tokens) }}</div>
      </div>
      <div class="bg-white border border-gray-200/80 rounded-2xl p-4 shadow-xs">
        <div class="text-[11px] text-gray-500">输出 Tokens</div>
        <div class="text-xl font-bold text-indigo-600 mt-1">{{ fmt(totals?.completion_tokens) }}</div>
      </div>
      <div class="bg-white border border-gray-200/80 rounded-2xl p-4 shadow-xs">
        <div class="text-[11px] text-gray-500">思考 Tokens</div>
        <div class="text-xl font-bold text-purple-600 mt-1">{{ fmt(totals?.reasoning_tokens) }}</div>
      </div>
      <div class="bg-white border border-gray-200/80 rounded-2xl p-4 shadow-xs">
        <div class="text-[11px] text-gray-500">缓存 Tokens</div>
        <div class="text-xl font-bold text-teal-600 mt-1">{{ fmt(totals?.cached_tokens) }}</div>
      </div>
      <div class="bg-white border border-gray-200/80 rounded-2xl p-4 shadow-xs">
        <div class="text-[11px] text-gray-500">总 Tokens</div>
        <div class="text-xl font-bold text-gray-900 mt-1">{{ fmt(totals?.total_tokens) }}</div>
      </div>
    </div>

    <!-- Model usage table -->
    <div class="bg-white border border-gray-200/80 rounded-2xl shadow-xs overflow-hidden">
      <div class="px-5 py-4 border-b border-gray-100 flex items-center justify-between">
        <h3 class="font-semibold text-gray-900 text-sm">模型用量明细</h3>
        <span class="text-xs text-gray-400">点击行展开按账号用量</span>
      </div>

      <div v-if="loading" class="px-5 py-16 text-center text-sm text-gray-400">加载中…</div>

      <div v-else-if="modelRows.length === 0" class="px-5 py-16 flex flex-col items-center gap-2 text-gray-400">
        <Inbox class="w-8 h-8" />
        <span class="text-sm">暂无调用数据,发起一次请求后这里会出现统计</span>
      </div>

      <div v-else class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead>
            <tr class="text-left text-xs text-gray-500 bg-gray-50/80">
              <th class="px-4 py-2.5 font-medium">模型</th>
              <th class="px-4 py-2.5 font-medium text-right">请求</th>
              <th class="px-4 py-2.5 font-medium text-right">成功</th>
              <th class="px-4 py-2.5 font-medium text-right">429</th>
              <th class="px-4 py-2.5 font-medium text-right">错误</th>
              <th class="px-4 py-2.5 font-medium text-right">输入</th>
              <th class="px-4 py-2.5 font-medium text-right">输出</th>
              <th class="px-4 py-2.5 font-medium text-right">思考</th>
              <th class="px-4 py-2.5 font-medium text-right">总Tokens</th>
              <th class="px-4 py-2.5 font-medium">最近使用</th>
              <th class="w-10"></th>
            </tr>
          </thead>
          <tbody>
            <template v-for="[name, model] in modelRows" :key="name">
              <tr
                class="border-t border-gray-100 hover:bg-gray-50/60 cursor-pointer"
                @click="hasAccountBreakdown(model) && toggleModel(name)"
              >
                <td class="px-4 py-2.5 font-medium text-gray-900 whitespace-nowrap">{{ name }}</td>
                <td class="px-4 py-2.5 text-right">{{ fmt(model.requests) }}</td>
                <td class="px-4 py-2.5 text-right text-emerald-600">{{ fmt(model.success) }}</td>
                <td class="px-4 py-2.5 text-right" :class="model.rate_limited > 0 ? 'text-amber-600' : 'text-gray-500'">
                  {{ fmt(model.rate_limited) }}
                </td>
                <td class="px-4 py-2.5 text-right" :class="model.errors > 0 ? 'text-red-600' : 'text-gray-500'">
                  {{ fmt(model.errors) }}
                </td>
                <td class="px-4 py-2.5 text-right text-gray-600">{{ fmt(model.prompt_tokens) }}</td>
                <td class="px-4 py-2.5 text-right text-gray-600">{{ fmt(model.completion_tokens) }}</td>
                <td class="px-4 py-2.5 text-right text-gray-600">{{ fmt(model.reasoning_tokens) }}</td>
                <td class="px-4 py-2.5 text-right font-semibold">{{ fmt(model.total_tokens) }}</td>
                <td class="px-4 py-2.5 text-xs text-gray-500 whitespace-nowrap">
                  {{ model.last_used ? formatDate(model.last_used) : '-' }}
                </td>
                <td class="px-2 py-2.5">
                  <component
                    :is="expandedModels.has(name) ? ChevronDown : ChevronRight"
                    v-if="hasAccountBreakdown(model)"
                    class="w-4 h-4 text-gray-400"
                  />
                </td>
              </tr>
              <tr
                v-if="expandedModels.has(name) && hasAccountBreakdown(model)"
                class="border-t border-gray-100 bg-gray-50/40"
              >
                <td :colspan="11" class="px-4 py-3">
                  <div class="overflow-x-auto">
                    <table class="w-full text-xs">
                      <thead>
                        <tr class="text-left text-gray-500">
                          <th class="px-3 py-1.5 font-medium">账号</th>
                          <th class="px-3 py-1.5 font-medium">等级</th>
                          <th class="px-3 py-1.5 font-medium text-right">今日请求</th>
                          <th class="px-3 py-1.5 font-medium text-right">今日成功</th>
                          <th class="px-3 py-1.5 font-medium text-right">今日429</th>
                          <th class="px-3 py-1.5 font-medium text-right">今日Tokens</th>
                          <th class="px-3 py-1.5 font-medium text-right">配额用量</th>
                          <th class="px-3 py-1.5 font-medium">状态</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr
                          v-for="acc in model.account_breakdown"
                          :key="acc.account_id"
                          class="border-t border-gray-200/60"
                        >
                          <td class="px-3 py-1.5 text-gray-800">
                            {{ acc.name }}
                            <span v-if="acc.email" class="text-gray-400">· {{ acc.email }}</span>
                          </td>
                          <td class="px-3 py-1.5">
                            <span class="px-1.5 py-0.5 rounded text-[10px] font-medium bg-blue-50 text-blue-700">
                              {{ acc.tier }}
                            </span>
                          </td>
                          <td class="px-3 py-1.5 text-right">{{ fmt(acc.today_requests) }}</td>
                          <td class="px-3 py-1.5 text-right text-emerald-600">{{ fmt(acc.today_success) }}</td>
                          <td class="px-3 py-1.5 text-right text-amber-600">{{ fmt(acc.today_rate_limited) }}</td>
                          <td class="px-3 py-1.5 text-right">{{ fmt(acc.today_tokens) }}</td>
                          <td class="px-3 py-1.5 text-right">{{ acc.quota_used_percent.toFixed(1) }}%</td>
                          <td class="px-3 py-1.5">
                            <span
                              class="px-1.5 py-0.5 rounded text-[10px] font-medium"
                              :class="acc.is_available ? 'bg-emerald-50 text-emerald-700' : 'bg-red-50 text-red-600'"
                            >
                              {{ acc.is_available ? '可用' : '冷却中' }}
                            </span>
                          </td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </td>
              </tr>
            </template>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>
