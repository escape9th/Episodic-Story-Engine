import os
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from zhipuai import ZhipuAI
from dotenv import load_dotenv

# 1. 加载配置
load_dotenv()
api_key = os.getenv("ZHIPU_API_KEY")

# 2. 初始化智谱客户端 (直连国内服务器，无需代理)
client = ZhipuAI(api_key=api_key)

app = FastAPI()

# 3. 允许跨域请求 (让 index.html 能顺利访问后端)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. 路由设置：访问首页
@app.get("/")
async def read_index():
    return FileResponse('index.html')

# 5. 定义前端发来的数据格式
class StoryPayload(BaseModel):
    worldview: str
    characters: str
    main_plot: str
    explicit_setting: str
    hidden_setting: str
    style_desc: str
    theme: str
    global_influence: str
    foreshadowing: str
    inner_meaning: str
    output_rules: str
    character_growth: str

# 6. 核心生成逻辑
@app.post("/generate_chapter")
async def generate_chapter(data: StoryPayload):
    print(f"\n🚀 [{time.strftime('%H:%M:%S')}] 接收到前端请求...")
    print(f"📝 剧情摘要: {data.main_plot[:20]}...")
    
    try:
        # 组装发送给 AI 的 Prompt
        master_prompt = f"""
你是顶级的单元剧小说家，擅长 Cyber-Folklore（赛博民俗）风格。
请根据以下设定创作：
【世界观】：{data.worldview}
【核心暗线】：{data.hidden_setting}（需通过氛围渗透，严禁直白写出）
【当前主线】：{data.main_plot}
【人物成长】：{data.character_growth}
【文风要求】：{data.style_desc}
【强制规则】：{data.output_rules}

请开始创作正文：
        """
        
        print("⏳ 正在请求智谱 GLM-4-Flash (免费版)，请稍候...")
        start_time = time.time()

        # 调用智谱 API
        response = client.chat.completions.create(
            model="glm-4-flash",  # 确保使用免费模型
            messages=[
                {"role": "system", "content": "你是一个专业的小说创作引擎。"},
                {"role": "user", "content": master_prompt}
            ],
            top_p=0.7,
            temperature=0.9,
            stream=False,
        )
        
        end_time = time.time()
        content = response.choices[0].message.content
        
        if content:
            print(f"✅ 生成成功！耗时: {round(end_time - start_time, 1)}秒")
            return {"status": "success", "content": content}
        else:
            return {"status": "error", "content": "AI 返回内容为空。"}
            
    except Exception as e:
        error_msg = str(e)
        print(f"❌ 运行报错: {error_msg}")
        return {"status": "error", "content": f"后端报错: {error_msg}"}

if __name__ == "__main__":
    import uvicorn
    print("🔥 小说生成器后端已启动！")
    print("👉 请在浏览器访问: http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)