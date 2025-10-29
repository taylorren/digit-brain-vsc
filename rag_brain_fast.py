import os
from dotenv import load_dotenv
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import time

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# 配置 Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:1.7b"

# 配置 DeepSeek
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

# ⚡ Performance optimization: Load model and index only once
print("⚡ 快速启动优化版 - 加载模型中...")
start_time = time.time()

# Load the local BAAI/bge-large-zh model for better Chinese support (1024-dim)
model = SentenceTransformer(os.path.join(os.path.dirname(__file__), 'models', 'bge-large-zh'))

# Load FAISS index and metadata
index = faiss.read_index(os.path.join(os.path.dirname(__file__), 'md_faiss.index'))
with open(os.path.join(os.path.dirname(__file__), 'md_faiss_meta.json'), 'r', encoding='utf-8') as f:
    meta = json.load(f)

load_time = time.time() - start_time
print(f"✅ 模型加载完成，耗时: {load_time:.2f}s")

def search(query, top_k=15):
    """优化的搜索函数"""
    # ⚡ 直接编码，关闭进度条以提升速度
    query_vec = model.encode([query], show_progress_bar=True)
    D, I = index.search(np.array(query_vec).astype('float32'), top_k)
    
    # ⚡ 使用列表推导式优化性能
    results = [
        {
            'score': float(score),
            'path': meta[idx]['path'],
            'content': meta[idx]['content'][:800]
        }
        for idx, score in zip(I[0], D[0])
    ]
    
    return results


def rag_ask(query, top_k=15, use_deepseek=False):
    """
    优化版RAG问答函数
    """
    start_time = time.time()
    print("[1/3] 🔍 正在检索相关片段...")
    
    docs = search(query, top_k)
    search_time = time.time() - start_time
    print(f"[2/3] 📄 已检索到{len(docs)}个片段，正在组织提示词... (检索耗时: {search_time:.2f}s)")
    
    # 相似度分析和过滤
    best_score = docs[0]['score'] if docs else float('inf')
    worst_score = docs[-1]['score'] if docs else float('inf')
    
    # 相似度质量评估 (L2距离：数值越小越相似)
    if best_score > 12:
        print(f"⚠️  相似度警告：最佳匹配度为 {best_score:.3f}，相关性较低")
    elif best_score < 5:
        print(f"✅ 高质量匹配：最佳相似度 {best_score:.3f}")
    
    # 显示检索到的相关文档（完整列表，已按相关度排序）
    print(f"\n📚 检索到的相关文档（共{len(docs)}个，按相关度排序，相似度范围: {best_score:.3f} - {worst_score:.3f}）:")
    for i, d in enumerate(docs, 1):
        file_name = os.path.basename(d['path'])
        file_ext = os.path.splitext(file_name)[1].upper()
        
        # 文件类型图标
        icon = {"MD": "📄", "PPTX": "🎯", "PDF": "📋"}.get(file_ext, "📁")
        
        # 相似度颜色标识
        if d['score'] < 8:
            score_indicator = "🟢"  # 高相关
        elif d['score'] < 12:
            score_indicator = "🟡"  # 中等相关
        else:
            score_indicator = "🔴"  # 低相关
        
        print(f"  #{i} {icon} {file_name} {score_indicator} (相似度: {d['score']:.3f})")
    print("")
    
    # 智能过滤文档
    if best_score > 15:
        filtered_docs = docs[:3]
        print(f"🔍 由于相关性较低，只使用前{len(filtered_docs)}个最相关文档")
    elif best_score > 10:
        filtered_docs = docs[:6]
    else:
        filtered_docs = docs
    
    # ⚡ 优化：使用列表推导式构建上下文
    context_parts = [
        f"【片段{i}】\n来源文件：{os.path.basename(d['path'])}\n内容：{d['content']}"
        for i, d in enumerate(filtered_docs, 1)
    ]
    context = "\n\n".join(context_parts)
    
    # 强化提示词
    prompt = (
        "你是任老师，请基于以下资料片段回答问题。\n\n"
        "**严格要求（必须遵守）:**\n"
        "1. 使用'任老师认为'、'任老师的观点是'等表达方式\n"
        "2. 每个观点后**必须**用方括号标注来源，格式：[来源：文件名.扩展名]\n"
        "3. 注意资料包含多种格式：.md（Markdown）、.pptx（PowerPoint）、.pdf（PDF）\n"
        "4. **优先引用相关度最高的片段**，无论是什么文件格式\n"
        "5. 不要输出<think>标记\n"
        "6. 确保每个要点都有明确的来源标注\n\n"
        "资料片段：\n"
        f"{context}\n\n"
        f"问题：{query}\n\n"
        "请按要求回答，每个观点都要标注来源，特别注意引用PPT和PDF文件内容："
    )
    
    if use_deepseek:
        return _call_deepseek(prompt)
    else:
        return _call_ollama(prompt, docs)


def _call_deepseek(prompt):
    """调用DeepSeek AI"""
    if not DEEPSEEK_API_KEY:
        return "[DeepSeek API Key 未设置，请设置环境变量 DEEPSEEK_API_KEY]"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    data = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": "你是一个有用的AI助手。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.6,
        "stream": False
    }
    try:
        ai_start = time.time()
        print("[3/3] 🤖 正在等待DeepSeek AI生成回答...")
        resp = requests.post(DEEPSEEK_URL, headers=headers, json=data, timeout=120)
        resp.raise_for_status()
        result = resp.json()
        ai_time = time.time() - ai_start
        print(f"⚡ AI回答生成耗时: {ai_time:.2f}s")
        
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "[DeepSeek未返回内容]")
        return content.strip()
    except Exception as e:
        return f"[DeepSeek调用失败: {e}]"


def _call_ollama(prompt, docs):
    """调用Ollama本地模型"""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    try:
        ai_start = time.time()
        print("[3/3] 🤖 正在等待Ollama生成回答...")
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        result = resp.json()
        ai_time = time.time() - ai_start
        print(f"⚡ AI回答生成耗时: {ai_time:.2f}s")
        
        response = result.get("response", "[Ollama未返回内容]")
        
        # 清理响应
        import re
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        
        # 检查来源标注
        if not re.search(r'\[来源：.*?\]', response) and not re.search(r'\(.*\.md\)', response):
            print("⚠️  检测到回答缺少来源标注，自动添加...")
            source_list = [os.path.basename(d['path']) for d in docs[:3]]
            response += f"\n\n**参考来源:** {', '.join(source_list)}"
        
        return response.strip()
    except Exception as e:
        return f"[Ollama调用失败: {e}]"


if __name__ == '__main__':
    use_deepseek = False
    print("\n🧠 数字大脑 - 性能优化版 ⚡")
    print("✨ 优化：快速启动、性能监控、智能过滤")
    print("\n命令说明:")
    print("  deepseek - 切换到DeepSeek AI")
    print("  ollama   - 切换到Ollama本地模型")
    print("  exit     - 退出程序")
    
    while True:
        query = input('\n💭 请输入你的问题: ')
        if query.lower() == 'exit':
            print("👋 再见！")
            break
        if query.lower() == 'deepseek':
            use_deepseek = True
            print("✅ 已切换到DeepSeek AI模式")
            continue
        if query.lower() == 'ollama':
            use_deepseek = False
            print("✅ 已切换到Ollama本地模式")
            continue
            
        if query.strip():
            total_start = time.time()
            print('\n🚀 开始处理...')
            answer = rag_ask(query, use_deepseek=use_deepseek)
            total_time = time.time() - total_start
            print(f'\n📝 【任老师的回答】\n{answer}\n')
            print(f"⚡ 总耗时: {total_time:.2f}s")
            print("-" * 50)