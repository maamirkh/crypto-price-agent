import os
import datetime
import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool
from agents.run import RunConfig
import httpx

# Load environment variables
load_dotenv()

# Load Gemini API key
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is missing in .env file.")

# Setup Gemini-compatible OpenAI client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Configure model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

# Runner configuration
config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# ✅ Tool 1: Fetch single or multiple prices
@function_tool
async def get_crypto_price(symbols: str) -> str:
    """
    Accepts one or multiple comma-separated symbols (e.g., BTCUSDT, ETHUSDT)
    and fetches their current prices from Binance.
    """
    url_base = "https://api.binance.com/api/v3/ticker/price"
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    results = []

    async with httpx.AsyncClient() as client:
        for symbol in symbol_list:
            try:
                url = f"{url_base}?symbol={symbol}"
                response = await client.get(url)
                data = response.json()

                if response.status_code == 200 and "price" in data:
                    price = float(data["price"])
                    results.append(f"{symbol}: ${price:,.8f}")
                else:
                    results.append(f"{symbol}: ❌ Invalid or not found")

            except Exception as e:
                results.append(f"{symbol}: ❌ Error - {str(e)}")

    return "Prices:\n" + "\n".join(results)

# ✅ Tool 2: List all active Binance trading pairs
@function_tool
async def list_all_binance_symbols() -> str:
    """
    Fetch all active trading symbols from Binance and return a sample list.
    """
    url = "https://api.binance.com/api/v3/exchangeInfo"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            data = response.json()

        symbols = [s["symbol"] for s in data["symbols"] if s["status"] == "TRADING"]
        total = len(symbols)
        sample = ", ".join(symbols[:20])

        return f"Total trading pairs: {total}\nSample: {sample}, ..."
    
    except Exception as e:
        return f"Error fetching symbols: {str(e)}"

# ✅ Define the agent with just two tools
agent = Agent(
    name="Crypto Smart Agent",
    instructions=(
        "You are a helpful assistant that can fetch crypto prices from Binance. "
        "You can fetch one or multiple symbols, and list all trading pairs."
    ),
    model=model,
    tools=[get_crypto_price, list_all_binance_symbols],
)

# ✅ Main interaction loop
async def main():
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        result = await Runner.run(agent, user_input, run_config=config)

        # Print latest model response
        for item in reversed(result.new_items):
            try:
                print("\nAssistant:", item.raw_item.content[0].text)
                break
            except Exception:
                continue

if __name__ == "__main__":
    asyncio.run(main())


