# Dogfood QA Report

**Target:** https://www.baidu.com/
**Date:** 2026-04-15
**Scope:** 百度首页桌面站小样本探索式测试：首页加载、顶部导航、搜索输入与提交主流程、文心助手入口、百度新闻入口。
**Tester:** Hermes Agent (automated exploratory QA)

---

## Executive Summary

| Severity | Count |
|----------|-------|
| 🔴 Critical | 0 |
| 🟠 High | 1 |
| 🟡 Medium | 2 |
| 🔵 Low | 0 |
| **Total** | **3** |

**Overall Assessment:** 百度首页与主要入口整体可用，但搜索主流程触发安全验证拦截，且搜索联想与返回链路存在可用性问题，影响无登录/自动化场景下的连续使用体验。

---

## Issues

### Issue #1: 首页搜索主流程被安全验证拦截，无法直接进入结果页

| Field | Value |
|-------|-------|
| **Severity** | High |
| **Category** | Functional |
| **URL** | https://www.baidu.com/ |

**Description:**
在首页输入测试关键词并提交后，未直接进入正常搜索结果页，而是被“百度安全验证”拦截。页面要求用户完成“拖动左侧滑块使图片为正”的图片旋转验证，导致标准搜索主流程中断。对于自动化代理、辅助技术用户或希望快速搜索的用户来说，这属于明显阻断。

**Steps to Reproduce:**
1. 打开 https://www.baidu.com/
2. 在首页搜索框输入“Hermes Agent dogfood 测试”
3. 按 Enter 提交搜索

**Expected Behavior:**
直接进入对应关键词的搜索结果页，用户可以继续浏览结果。

**Actual Behavior:**
页面跳转到“百度安全验证”，要求完成滑块旋转图片验证后才能继续，正常搜索结果未展示。

**Screenshot:**
Original screenshot captured during the source run (binary not committed in this repo).

**Console Errors** (if applicable):
```text
None observed.
```

---

### Issue #2: 搜索联想词与已输入查询明显不相关

| Field | Value |
|-------|-------|
| **Severity** | Medium |
| **Category** | UX |
| **URL** | https://www.baidu.com/ |

**Description:**
在首页搜索框输入“Hermes Agent dogfood 测试”后，下拉联想建议并未围绕完整查询或“dogfood 测试”意图展开，而是出现大量泛化的英文品牌/词条，如 “hermes tracking”、“hermes track”、“hermes trismegistus”等。这种联想结果与当前查询意图偏差较大，容易误导用户点击到无关搜索方向。

**Steps to Reproduce:**
1. 打开 https://www.baidu.com/
2. 在首页搜索框输入“Hermes Agent dogfood 测试”
3. 观察联想词下拉列表

**Expected Behavior:**
联想词应尽量贴近当前完整查询，或至少与“Agent / dogfood / 测试”意图相关。

**Actual Behavior:**
联想词主要围绕泛化的“Hermes”品牌/英文词条展开，与完整查询相关性较弱。

**Screenshot:**
Original screenshot captured during the source run (binary not committed in this repo).

**Console Errors** (if applicable):
```text
None observed.
```

---

### Issue #3: 从文心助手页使用返回操作未能回到百度首页，历史链路表现不稳定

| Field | Value |
|-------|-------|
| **Severity** | Medium |
| **Category** | Functional |
| **URL** | https://chat.baidu.com/?enter_type=home_operate |

**Description:**
从百度首页点击“复杂问题就找文心助手”进入文心助手页后，使用浏览器后退操作时，并未顺利回到百度首页，而是停留在文心相关页面。对用户来说，这会造成页面链路理解困难；对自动化工作流来说，也会增加 flow 恢复成本。

**Steps to Reproduce:**
1. 打开 https://www.baidu.com/
2. 点击“复杂问题就找文心助手，深入思考回答更优”入口
3. 在文心助手页面执行浏览器后退

**Expected Behavior:**
后退应返回原始百度首页。

**Actual Behavior:**
后退后仍停留在文心相关页面，未恢复到首页，需要重新导航到百度首页。

**Screenshot:**
Original screenshot captured during the source run (binary not committed in this repo).

**Console Errors** (if applicable):
```text
None observed.
```

---

## Issues Summary Table

| # | Title | Severity | Category | URL |
|---|-------|----------|----------|-----|
| 1 | 首页搜索主流程被安全验证拦截，无法直接进入结果页 | High | Functional | https://www.baidu.com/ |
| 2 | 搜索联想词与已输入查询明显不相关 | Medium | UX | https://www.baidu.com/ |
| 3 | 从文心助手页使用返回操作未能回到百度首页，历史链路表现不稳定 | Medium | Functional | https://chat.baidu.com/?enter_type=home_operate |

## Testing Coverage

### Pages Tested
- 百度首页（https://www.baidu.com/）
- 百度安全验证页（搜索后触发）
- 文心助手入口页 / 对话页（https://chat.baidu.com/）
- 百度新闻页（http://news.baidu.com）

### Features Tested
- 首页加载与视觉检查
- 浏览器 console 基础检查
- 首页搜索输入与提交
- 搜索联想词观察
- 文心助手入口跳转
- 文心助手单轮提问与回答返回
- 顶部“新闻”导航入口跳转

### Not Tested / Out of Scope
- 登录流程
- 图片、视频、地图、贴吧、网盘、文库等其余顶部入口的深入测试
- 首页“设置”菜单展开行为
- 热搜条目逐条点击验证
- 移动端布局与响应式行为
- 安全验证滑块的人工完成与验证后结果页质量

### Blockers
- 搜索主流程被百度安全验证拦截，无法在当前会话中继续检查正常搜索结果页的相关性、结果布局与分页链路。

---

## Notes

1. 百度首页本身在首屏加载、布局和视觉呈现上表现稳定，未见明显白屏、JS 报错或布局错位。
2. 文心助手入口可正常打开，且无需登录即可完成单轮问答，这一入口的可用性较好。
3. 百度新闻页正常打开，说明顶部导航至少部分入口工作正常。
4. 本次最主要的问题集中在“主搜索流程被风控打断”和“返回链路不稳定”，这两点对真实 end-to-end 体验影响最大。
