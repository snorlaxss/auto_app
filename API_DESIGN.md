# 机器人遥操可视化系统 — 后端 API 设计方案

## 1. 总体架构

```
┌──────────────────────────┐       ┌──────────────────────────────────┐
│  前端 (Next.js)           │       │          后端服务                   │
│                           │       │                                  │
│  ┌─────────────────┐     │  REST │  ┌────────────────────────────┐  │
│  │ 控制指令 (采集/抓取) │───▶│────▶│  POST /api/v1/capture         │  │
│  │  /停止/复位)      │◀───│──────│  POST /api/v1/grasp           │  │
│  └─────────────────┘     │      │  POST /api/v1/stop             │  │
│                           │      │  POST /api/v1/home             │  │
│                           │      │  POST /api/v1/bringup/*        │  │
│                           │      │  GET  /api/v1/status           │  │
│  ┌─────────────────┐     │      └────────────────────────────┘  │
│  │ 实时数据消费       │◀───│──────┌────────────────────────────┐  │
│  │ (关节状态)         │     │ WebSocket                           │  │
│  └─────────────────┘     │      │  ws://<host>/ws/robot/state    │  │
│                           │      └────────────────────────────┘  │
│  ┌─────────────────┐     │                                        │
│  │ 操作结果轮询      │◀───│──────▶ GET /api/v1/operation/{id}     │  │
│  └─────────────────┘     │  REST  (每 500ms 轮询直到完成)         │  │
└──────────────────────────┘       └──────────────────────────────────┘
```

**协议选择理由：**

| 通道 | 协议 | 理由 |
|------|------|------|
| 指令下发（采集/抓取/停止） | REST over HTTP | 一次性请求-响应，语义明确 |
| 图像采集 | REST over HTTP | `POST /capture` 直接返回 base64 图片或 URL，点一次拿一张 |
| 操作事件 | REST 轮询 | `GET /operation/{id}` 每 500ms 查询，操作只 2-5 秒 |
| 关节状态推送 | WebSocket | 高频实时数据（20-100 Hz），需要服务端主动推送，是唯一需要 WS 的通道 |

---

## 2. REST API

Base URL: `http://<host>:8080/api/v1`

所有 REST 响应使用统一信封：

```json
{
  "code": 0,
  "message": "success",
  "data": { ... }
}
```

错误码约定：

| code | 含义 |
|------|------|
| 0 | 成功 |
| 1001 | 参数错误 |
| 1002 | 机器人忙（正在执行其他操作） |
| 1003 | 机器人未连接 / 离线 |
| 1004 | 相机未连接 / 离线 |
| 1005 | 急停已触发 |
| 2001 | 内部错误 |

---

### 2.1 采集图像

触发相机拍摄一帧，直接返回 base64 编码的图片数据，前端拿到即可 `<img src="data:image/jpeg;base64,...">` 显示，一次 HTTP 往返完成。

```
POST /api/v1/capture
```

**Request:** `Content-Type: application/json`

```json
{
  "camera_id": "cam_01",
  "width": 1280,
  "height": 720,
  "format": "jpeg",
  "quality": 85
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| camera_id | string | 否 | 相机 ID，默认 `cam_01` |
| width | int | 否 | 图像宽度，默认 1280 |
| height | int | 否 | 图像高度，默认 720 |
| format | string | 否 | 编码格式：`jpeg` / `png`，默认 `jpeg` |
| quality | int | 否 | JPEG 质量 1-100，默认 85 |

**Response `200`:**

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "image_id": "img_20260610_143021_001",
    "timestamp": 1718005821.234,
    "width": 1280,
    "height": 720,
    "format": "jpeg",
    "size_bytes": 245760,
    "image_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBD...",
    "image_url": null
  }
}
```

> **说明：** 默认返回 `image_base64`，前端直接用 `<img src="data:image/jpeg;base64,${image_base64}">` 显示，一次请求搞定。如果图片较大需要持久化，可传 `"return_mode": "url"` 改用 `image_url` 间接引用。

**Response `400/409` (失败):**

```json
{
  "code": 1004,
  "message": "相机未连接",
  "data": null
}
```

---

### 2.2 开始抓取

```
POST /api/v1/grasp
```

**Request:**

```json
{
  "target_object": "object_01",
  "approach_speed": 0.5,
  "grasp_force": 0.8,
  "trajectory_mode": "linear"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| target_object | string | 否 | 目标物体 ID |
| approach_speed | float | 否 | 接近速度 0.1-1.0，默认 0.5 |
| grasp_force | float | 否 | 抓取力 0.1-1.0，默认 0.8 |
| trajectory_mode | string | 否 | 轨迹模式：`linear` / `arc`，默认 `linear` |

**Response `200`:**

```json
{
  "code": 0,
  "message": "抓取操作已启动",
  "data": {
    "operation_id": "op_20260610_143022_001",
    "estimated_duration_ms": 4000
  }
}
```

**Response `409` (机器人忙):**

```json
{
  "code": 1002,
  "message": "机器人正在执行操作中，请先停止或等待完成",
  "data": null
}
```

---

### 2.3 停止 / 复位

```
POST /api/v1/stop
```

**Request:**

```json
{
  "mode": "reset",
  "emergency": false
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| mode | string | 否 | `reset`（回到初始位姿）/ `hold`（保持当前位置），默认 `reset` |
| emergency | bool | 否 | 是否紧急停止（忽略当前轨迹直接刹车），默认 `false` |

**Response `200`:**

```json
{
  "code": 0,
  "message": "停止指令已发送",
  "data": {
    "operation_id": "op_20260610_143025_001",
    "mode": "reset"
  }
}
```

---

### 2.4 回 Home

对应网页端 `Home` 按钮，向当前 Marvin 动作端发布双臂 Home；如果动作端未启动，则使用主 ROS2 节点兜底发布。

```
POST /api/v1/home
```

**Request:** 无请求体。

**Response `200`:**

```json
{
  "code": 0,
  "message": "Home指令已发送",
  "data": {
    "operation_id": "op_20260713_091500_ab12cd",
    "source": "robotaction_node"
  }
}
```

`source` 取值：

| 值 | 说明 |
|----|------|
| `robotaction_node` | 通过 Marvin 动作端发布 Home |
| `ros2_node` | 动作端未启动，通过主 ROS2 节点兜底发布 Home |

---

### 2.5 机器人 Bringup 控制

这些接口对应网页端 bringup 按钮。后端通过：

```bash
/home/snorlax/work/distri_0112_org/ros2_ws/start_marvin_tmux.sh
```

执行 tmux 启动/重启命令。启动接口会设置 `TMUX_ATTACH=0`，不会阻塞 HTTP 请求。

#### 2.5.1 启动机器人 Bringup

```
POST /api/v1/bringup/start
```

执行：

```bash
TMUX_ATTACH=0 bash start_marvin_tmux.sh start
```

**Response `200`:**

```json
{
  "code": 0,
  "message": "机器人Bringup启动指令已发送",
  "data": {
    "action": "bringup_start",
    "status": "completed",
    "command": ["bash", "/home/snorlax/work/distri_0112_org/ros2_ws/start_marvin_tmux.sh", "start"],
    "output": "tmux session 已后台运行: marvin_bringup"
  }
}
```

#### 2.5.2 重发 Control 初始化

```
POST /api/v1/bringup/restart-control
```

执行：

```bash
bash start_marvin_tmux.sh restart control
```

会重新发送：

```bash
ros2 service call /control/set_ready std_srvs/srv/Trigger "{}"
ros2 service call /control/set_mode marvin_msgs/srv/Int "{data: 3}"
ros2 topic pub -1 /control/gripL std_msgs/msg/Bool "{data: True}"
ros2 topic pub -1 /control/gripR std_msgs/msg/Bool "{data: True}"
```

#### 2.5.3 重启 Planner

```
POST /api/v1/bringup/restart-planner
```

执行：

```bash
bash start_marvin_tmux.sh restart planner
```

**错误响应示例：**

```json
{
  "code": 1003,
  "message": "Marvin tmux命令失败: restart planner\nreturncode=1\n错误: tmux session 不存在: marvin_bringup",
  "data": null
}
```

---

### 2.6 查询状态

```
GET /api/v1/status
```

**Response `200`:**

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "robot": {
      "connected": true,
      "state": "idle",
      "joint_count": 6,
      "current_operation": null
    },
    "camera": {
      "connected": true,
      "streaming": true,
      "fps": 30
    },
    "safety": {
      "e_stop_active": false,
      "collision_detected": false,
      "in_safe_zone": true
    },
    "last_updated": 1718005823.456
  }
}
```

`robot.state` 枚举：

| 值 | 说明 |
|----|------|
| `idle` | 待机，无操作 |
| `moving` | 正在运动（抓取/复位） |
| `holding` | 保持当前位置（暂停） |
| `error` | 错误状态 |
| `offline` | 未连接 |

---

## 3. WebSocket 实时通道

### 3.1 连接参数

所有 WebSocket 连接 URL 支持 query 参数：

```
ws://<host>:8080/ws/<channel>?interval_ms=20&compression=1
```

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| interval_ms | int | 20 | 推送间隔（毫秒），最小值 10 |
| compression | int | 1 | 0=不压缩, 1=permessage-deflate |

### 3.2 认证

WebSocket 首次连接时在 URL 中携带 token（或连接后首条消息认证）：

```
ws://<host>:8080/ws/robot/state?token=<jwt_token>
```

---

### 3.3 通道一：机器人关节状态 `/ws/robot/state`

**服务端 → 客户端 推送频率：** 20-100 Hz（可配置）

**消息格式（双向 JSON）：**

```json
{
  "type": "joint_state",
  "seq": 12345,
  "timestamp": 1718005823.456,
  "robot_id": "arm_6dof_01",
  "joints": [
    {
      "name": "joint_1",
      "angle_rad": 0.5236,
      "velocity_rad_s": 0.12,
      "torque_nm": 1.5,
      "temperature_c": 42.3,
      "status": "ok"
    },
    {
      "name": "joint_2",
      "angle_rad": -1.0472,
      "velocity_rad_s": -0.08,
      "torque_nm": 2.1,
      "temperature_c": 38.7,
      "status": "ok"
    },
    {
      "name": "joint_3",
      "angle_rad": 0.7854,
      "velocity_rad_s": 0.0,
      "torque_nm": 0.0,
      "temperature_c": 35.1,
      "status": "ok"
    },
    {
      "name": "joint_4",
      "angle_rad": -0.3491,
      "velocity_rad_s": 0.05,
      "torque_nm": 0.8,
      "temperature_c": 40.2,
      "status": "ok"
    },
    {
      "name": "joint_5",
      "angle_rad": 1.5708,
      "velocity_rad_s": 0.0,
      "torque_nm": 0.0,
      "temperature_c": 36.5,
      "status": "ok"
    },
    {
      "name": "joint_6",
      "angle_rad": 0.0,
      "velocity_rad_s": 0.0,
      "torque_nm": 0.0,
      "temperature_c": 33.8,
      "status": "ok"
    }
  ],
  "end_effector": {
    "pose": {
      "x_m": 0.45,
      "y_m": 0.12,
      "z_m": 0.38,
      "roll_rad": 0.0,
      "pitch_rad": 1.5708,
      "yaw_rad": 0.5236
    },
    "gripper_open_mm": 25.0,
    "wrench": {
      "fx_n": 0.0,
      "fy_n": 0.0,
      "fz_n": -2.3
    }
  }
}
```

关键字段说明：

| 字段 | 说明 |
|------|------|
| `joints[].angle_rad` | 关节角度（弧度），URDF 模型直接驱动 |
| `joints[].velocity_rad_s` | 关节角速度 |
| `joints[].torque_nm` | 关节力矩 |
| `joints[].status` | `ok` / `warning` / `error` — 驱动前端指示灯 |
| `end_effector.pose` | 末端执行器在世界坐标系下的位姿 |
| `end_effector.gripper_open_mm` | 夹爪开度 |

---

### 2.7 查询操作结果（轮询）

操作事件只有"开始"和"结束"两条消息，WebSocket 过重。采用 REST 轮询：`POST /grasp` 返回 `operation_id`，前端定时查询 `GET /operation/{id}` 直到状态变为终态。

```
GET /api/v1/operation/{operation_id}
```

**Response `200` (进行中):**

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "operation_id": "op_20260610_143022_001",
    "operation_type": "grasp",
    "status": "running",
    "created_at": 1718005822.000,
    "updated_at": 1718005823.100
  }
}
```

**Response `200` (已完成):**

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "operation_id": "op_20260610_143022_001",
    "operation_type": "grasp",
    "status": "completed",
    "result": "success",
    "duration_ms": 3850,
    "created_at": 1718005822.000,
    "updated_at": 1718005825.850
  }
}
```

**Response `200` (失败):**

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "operation_id": "op_20260610_143022_001",
    "operation_type": "grasp",
    "status": "failed",
    "result": "failed",
    "error_code": 2001,
    "error_message": "抓取失败：物体滑落",
    "duration_ms": 3200,
    "created_at": 1718005822.000,
    "updated_at": 1718005825.200
  }
}
```

`status` 枚举：

| 值 | 含义 | 是否终态 |
|----|------|----------|
| `running` | 执行中，继续轮询 | 否 |
| `completed` | 成功完成 | 是 |
| `failed` | 执行失败 | 是 |
| `aborted` | 被用户停止 | 是 |

**前端轮询逻辑：**

```
POST /grasp → 拿到 operation_id → 开始转圈
    │
    └─ 每 500ms: GET /operation/{id}
         ├─ status="running"   → 继续转圈，继续轮询
         ├─ status="completed" → 停止转圈，显示"已完成" ✓
         ├─ status="failed"    → 停止转圈，显示错误信息
         └─ status="aborted"   → 停止转圈，回 idle
```

轮询间隔建议 **500ms**，操作通常持续 2-5 秒，总共只会发出 4-10 次轻量请求，比维护一个 WebSocket 连接更简单。

**前端状态机：**

```
idle ──(POST /grasp)──▶ running（转圈 + 轮询）
                          │
              ┌─ status:"completed" ──▶ completed（✓ 已完成）
              │
              └─ status:"failed"    ──▶ 显示错误，回 idle

running ──(POST /stop)──▶ stopping（转圈 + 轮询）
                            │
                ┌─ status:"aborted" ──▶ idle
                │
                └─ status:"failed"  ──▶ 显示错误，回 idle
```

> **备选 — 长轮询 (long polling)：** 如果希望减少请求数，可以让 `GET /operation/{id}?timeout=30` 在没有状态变化时挂起连接最多 30s，状态变化时立即返回。效果接近 WebSocket 但无需升级协议。对于当前场景（操作 2-5 秒），短轮询已经足够。

---

## 4. 前端对接映射

当前 mock 行为与真实 API 的对应关系：

| 当前 Mock | 替换为 | 通道 |
|-----------|--------|------|
| `handleCapture` → `setTimeout(1400)` | `POST /api/v1/capture` → 拿到 `image_url` 后显示 | REST |
| `handleGrasp` → `setTimeout(4000)` 设 `completed` | `POST /api/v1/grasp` → 轮询 `GET /operation/{id}` 直到 `status:"completed"` | REST |
| `handleStop` → `setTimeout(2000)` 回 `idle` | `POST /api/v1/stop` → 轮询 `GET /operation/{id}` 直到 `status:"aborted"` | REST |
| `Home` 按钮 | `POST /api/v1/home` | REST |
| `启动机器人Bringup` 按钮 | `POST /api/v1/bringup/start` | REST |
| `重发Control初始化` 按钮 | `POST /api/v1/bringup/restart-control` | REST |
| `重启Planner` 按钮 | `POST /api/v1/bringup/restart-planner` | REST |
| 转圈动画 (`Loader2`) | 发起操作后开始转圈，轮询到终态后停止 | — |
| "已完成" 文字 + 绿色勾 | 轮询返回 `status:"completed"` + `result:"success"` 后展示 | — |
| 静态 `/urdf-robot.png` | WS `/ws/robot/state` 推送 `joints[].angle_rad`，前端驱动 3D 模型 | WebSocket |
| 静态 `/grasp-scene.png` | `POST /capture` 返回 `image_base64`，直接 `<img src="data:image/jpeg;base64,...">` | REST |
| `status` 本地 state | `GET /api/v1/status` 初始查询 + WS 事件驱动状态变更 | REST + WS |

---

## 5. 前端集成代码示例

```typescript
// lib/robot-api.ts — 后端通信层

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8080"
const WS_BASE  = API_BASE.replace(/^http/, "ws")

// ── REST ──────────────────────────────────────────────

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}/api/v1${path}`, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  })
  const json = await res.json()
  if (json.code !== 0) throw new ApiError(json.code, json.message)
  return json.data
}

export const api = {
  capture:      (params?: CaptureParams)  => request<CaptureResult>("POST", "/capture", params),
  grasp:        (params?: GraspParams)    => request<GraspResult>("POST", "/grasp", params),
  stop:         (params?: StopParams)     => request<StopResult>("POST", "/stop", params),
  home:         ()                        => request<HomeResult>("POST", "/home"),
  bringupStart: ()                        => request<BringupResult>("POST", "/bringup/start"),
  restartControl: ()                      => request<BringupResult>("POST", "/bringup/restart-control"),
  restartPlanner: ()                      => request<BringupResult>("POST", "/bringup/restart-planner"),
  getStatus:    ()                        => request<RobotStatus>("GET", "/status"),
  getOperation: (id: string)              => request<OperationResult>("GET", `/operation/${id}`),
}

// ── WebSocket ─────────────────────────────────────────

export function connectRobotState(onJointState: (s: JointState) => void): () => void {
  const ws = new WebSocket(`${WS_BASE}/ws/robot/state`)
  ws.onmessage = (e) => onJointState(JSON.parse(e.data))
  return () => ws.close()  // cleanup
}

// ── 操作轮询 ─────────────────────────────────────────

export async function pollUntilDone(
  operationId: string,
  onStatus: (status: string) => void,
  intervalMs = 500,
): Promise<OperationResult> {
  while (true) {
    const data = await api.getOperation(operationId)
    onStatus(data.status)
    if (data.status === "completed") return data
    if (data.status === "failed" || data.status === "aborted") return data
    await new Promise((r) => setTimeout(r, intervalMs))
  }
}

// 前端使用示例：
// const { operation_id } = await api.grasp()
// setStatus("running")                            // 开始转圈
// const result = await pollUntilDone(operation_id, () => {})
// if (result.status === "completed") {
//   setStatus("completed")                        // 停止转圈，显示"已完成"
// } else {
//   setStatus("idle")                             // 停止转圈，显示错误
// }

// 图像采集直接用 REST，无需 WebSocket：
// const { image_base64 } = await api.capture()
// setImageData(`data:image/jpeg;base64,${image_base64}`)  // 直接给 <img src>
```

> **渐进式接入策略：** 后端可以逐步实现——先接全部 REST 接口（采集/抓取/停止 + 操作轮询），就能 100% 替换模拟逻辑；之后再接入 `/ws/robot/state` 驱动 3D 模型实现实时关节可视化。

---

## 6. 部署参考

```
                    ┌──────────┐
                    │  Next.js  │  :3000 (开发) / static export (生产)
                    └─────┬────┘
                          │ REST :8080  /  WS :8080
                    ┌─────▼────┐
                    │  API 网关  │  (Nginx / Envoy，可选)
                    └─────┬────┘
              ┌───────────┼───────────┐
              │           │           │
        ┌─────▼────┐ ┌───▼────┐ ┌───▼────┐
        │ 机器人控制 │ │ 相机服务 │ │ 状态聚合 │
        │ (ROS2桥接)│ │ (GStreamer│ │ (Redis) │
        │          │ │  /v4l2) │ │        │
        └──────────┘ └────────┘ └────────┘
```

机器人侧建议通过 **ROS 2** 桥接：
- `/joint_states` topic → WebSocket 推送
- `/tf` → end_effector pose
- MoveIt 2 action server → grasp/stop 指令
- `image_transport` → 相机帧

---

## 7. 消息汇总速查表

| 方向 | 通道 | 内容 | 频率 |
|------|------|------|------|
| 前→后 | `POST /capture` | 触发拍照，直接返回 base64 图片 | 按需 |
| 前→后 | `POST /grasp` | 启动抓取，返回 operation_id | 按需 |
| 前→后 | `POST /stop` | 停止/复位，返回 operation_id | 按需 |
| 前→后 | `POST /home` | 发布双臂 Home | 按需 |
| 前→后 | `POST /bringup/start` | 启动 Marvin tmux bringup | 按需 |
| 前→后 | `POST /bringup/restart-control` | 重发 control 初始化 service/topic | 按需 |
| 前→后 | `POST /bringup/restart-planner` | 重启 planner pane | 按需 |
| 前→后 | `GET /status` | 查询机器人/相机/安全状态 | 按需 |
| 前→后 | `GET /operation/{id}` | 轮询操作结果（500ms 间隔） | 每条操作 4-10 次 |
| 后→前 | `WS /ws/robot/state` | 关节角度、末端位姿（唯一 WS 通道） | 20-100 Hz |
