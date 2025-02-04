import os
import discord
import asyncio
from dotenv import load_dotenv
from predict import (
    analyze_stock_detailed,
    analyze_crypto,
    generate_top10_stock_graphs_30min,
    generate_top25_stock_summary_30min,  # Summaries for top 25 stocks
    generate_top10_crypto_graphs_30min,
    generate_top25_crypto_summary_30min  # Summaries for top 25 cryptos
)

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DISCORD_CHANNEL_ID = int(os.getenv("DISCORD_CHANNEL_ID"))

client = discord.Client(intents=discord.Intents.default())
tree = discord.app_commands.CommandTree(client)

# Helper function to split text into chunks (Discord limit ~2000 characters per message)
def split_text(text, limit=2000):
    lines = text.split("\n")
    chunks = []
    current_chunk = ""
    for line in lines:
        if len(current_chunk) + len(line) + 1 > limit:
            chunks.append(current_chunk)
            current_chunk = line + "\n"
        else:
            current_chunk += line + "\n"
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

@client.event
async def on_ready():
    print(f"Logged in as {client.user}")
    try:
        synced = await tree.sync()
        print(f"Synced {len(synced)} command(s)")
    except Exception as e:
        print(f"Error syncing commands: {e}")
    
    # Send one update immediately on startup
    print("Sending initial update...")
    await send_updates()
    
    # Schedule periodic updates (first update after 30 minutes)
    client.loop.create_task(periodic_updates())

@tree.command(name="predict", description="Get live prediction for a given stock or crypto symbol")
async def predict(interaction: discord.Interaction, market: str, symbol: str):
    await interaction.response.defer()
    if market.lower() == "stock":
        summary, graphs = analyze_stock_detailed(symbol)
        if not graphs:
            await interaction.followup.send(content=summary)
            return
        embed = discord.Embed(
            title=f"{symbol.upper()} Stock Detailed Analysis",
            description=summary,
            color=0x00ff00
        )
        embed.set_image(url="attachment://price_trend.png")
        embed.set_footer(text="Charts: Price Trend, Daily % Change, Volume Trend")
        files = [
            discord.File(graphs[0], filename="price_trend.png"),
            discord.File(graphs[1], filename="daily_change.png"),
            discord.File(graphs[2], filename="volume_trend.png")
        ]
        await interaction.followup.send(embed=embed, files=files)
    elif market.lower() == "crypto":
        summary, graphs = analyze_crypto(symbol)
        if not graphs:
            await interaction.followup.send(content=summary)
            return
        embed = discord.Embed(
            title=f"{symbol.capitalize()} Crypto Detailed Analysis",
            description=summary,
            color=0x0099ff
        )
        embed.set_image(url="attachment://crypto_price.png")
        embed.set_footer(text="Charts: Price Trend, Daily % Change")
        files = [
            discord.File(graphs[0], filename="crypto_price.png"),
            discord.File(graphs[1], filename="crypto_daily_change.png")
        ]
        await interaction.followup.send(embed=embed, files=files)
    else:
        await interaction.followup.send(content="Market type must be 'stock' or 'crypto'.")

async def send_updates():
    channel = client.get_channel(DISCORD_CHANNEL_ID)
    if channel is None:
        print(f"Channel with ID {DISCORD_CHANNEL_ID} not found.")
        return

    # --- Send Top 10 Graphs for Stocks & Crypto ---
    stock_graphs = generate_top10_stock_graphs_30min()
    crypto_graphs = generate_top10_crypto_graphs_30min()

    if stock_graphs:
        embed_stock = discord.Embed(
            title="Live Top 10 Stocks Graphs (Past 30 Minutes)",
            color=0xffaa00
        )
        embed_stock.set_footer(text="Graphs: % Change and Current Price")
        try:
            await channel.send(embed=embed_stock, files=[
                discord.File(stock_graphs[0], filename="top10_stocks_pct_change.png"),
                discord.File(stock_graphs[1], filename="top10_stocks_price.png")
            ])
            print("Top 10 stock graphs sent.")
        except Exception as e:
            print(f"Error sending top 10 stock graphs: {e}")

    if crypto_graphs:
        embed_crypto = discord.Embed(
            title="Live Top 10 Cryptos Graphs (Past 30 Minutes)",
            color=0x00ffff
        )
        embed_crypto.set_footer(text="Graphs: % Change and Current Price")
        try:
            await channel.send(embed=embed_crypto, files=[
                discord.File(crypto_graphs[0], filename="top10_crypto_pct_change.png"),
                discord.File(crypto_graphs[1], filename="top10_crypto_price.png")
            ])
            print("Top 10 crypto graphs sent.")
        except Exception as e:
            print(f"Error sending top 10 crypto graphs: {e}")

    # --- Send Top 25 Summaries for Stocks & Crypto ---
    stock_summary = generate_top25_stock_summary_30min()
    crypto_summary = generate_top25_crypto_summary_30min()

    # Split summaries if they exceed Discord's message limit.
    stock_chunks = split_text(stock_summary)
    crypto_chunks = split_text(crypto_summary)

    try:
        await channel.send("**Live Top 25 Stocks Summary (Past 30 Minutes):**")
        for chunk in stock_chunks:
            await channel.send(chunk)
        print("Top 25 stock summary sent.")
    except Exception as e:
        print(f"Error sending top 25 stock summary: {e}")

    try:
        await channel.send("**Live Top 25 Cryptos Summary (Past 30 Minutes):**")
        for chunk in crypto_chunks:
            await channel.send(chunk)
        print("Top 25 crypto summary sent.")
    except Exception as e:
        print(f"Error sending top 25 crypto summary: {e}")

async def periodic_updates():
    # Wait 30 minutes before sending the first periodic update
    await asyncio.sleep(1800)
    while not client.is_closed():
        await send_updates()
        print("Periodic update sent.")
        await asyncio.sleep(1800)  # 30 minutes

client.run(DISCORD_TOKEN)