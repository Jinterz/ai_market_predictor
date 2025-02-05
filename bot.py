# bot.py
# This file runs the Discord bot for the AI Market Predictor.
# It includes commands to get live market analysis, historical predictions,
# news headlines, comparisons, and AI recommendations.
#
# The bot also streams top 25 stock and crypto summaries as individual embeds.

import os
import discord
import asyncio
from dotenv import load_dotenv
from pymongo import MongoClient

# Import necessary functions from our prediction module.
from predict import (
    analyze_stock_detailed,
    analyze_crypto,
    generate_top10_stock_graphs_30min,
    generate_top10_crypto_graphs_30min,
    generate_top25_crypto_summary_30min,  # Synchronous crypto summary (optional)
    get_ai_recommendation,
    async_generate_top25_stock_summaries,
    async_generate_top25_crypto_summaries  # New async generator for crypto summaries
)

# Load environment variables.
load_dotenv()

# Retrieve tokens and channel IDs from environment.
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DISCORD_CHANNEL_ID = int(os.getenv("DISCORD_CHANNEL_ID"))

# Set up the Discord client and command tree.
intents = discord.Intents.default()
client = discord.Client(intents=intents)
tree = discord.app_commands.CommandTree(client)

# Set up MongoDB connection.
MONGO_URI = os.getenv("MONGO_URI")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client['investment_db']
predictions_collection = db['predictions']

def split_text(text, limit=2000):
    """
    Split a long string into chunks under the specified character limit.
    Useful for splitting messages to stay within Discord's limits.
    """
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
    """Event triggered when the bot is ready."""
    print(f"Logged in as {client.user}")
    try:
        synced = await tree.sync()
        print(f"Synced {len(synced)} command(s)")
    except Exception as e:
        print(f"Error syncing commands: {e}")
    
    # Send an initial update to the designated channel.
    print("Sending initial update...")
    await send_updates()
    
    # Schedule periodic updates every 30 minutes.
    client.loop.create_task(periodic_updates())

@tree.command(name="predict", description="Get live analysis for a given stock or crypto symbol")
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
            color=0xFF3131
        )
        embed.set_footer(text="Charts: Price Trend, Daily % Change")
        # First, send the embed without images.
        await interaction.followup.send(embed=embed)
        # Then, send the image files as separate messages.
        await interaction.followup.send(file=discord.File(graphs[0], filename="price_trend.png"))
        await interaction.followup.send(file=discord.File(graphs[1], filename="daily_change.png"))
    elif market.lower() == "crypto":
        summary, graphs = analyze_crypto(symbol)
        if not graphs:
            await interaction.followup.send(content=summary)
            return
        embed = discord.Embed(
            title=f"{symbol.capitalize()} Crypto Detailed Analysis",
            description=summary,
            color=0xFF3131
        )
        embed.set_footer(text="Charts: Price Trend, Daily % Change")
        await interaction.followup.send(embed=embed)
        await interaction.followup.send(file=discord.File(graphs[0], filename="crypto_price.png"))
        await interaction.followup.send(file=discord.File(graphs[1], filename="daily_change.png"))
    else:
        await interaction.followup.send(content="Market type must be 'stock' or 'crypto'.")

@tree.command(name="history", description="Retrieve the last 10 market analyses")
async def history(interaction: discord.Interaction):
    """Command to retrieve the last 10 stored market analyses from MongoDB."""
    await interaction.response.defer()
    try:
        preds = list(predictions_collection.find({}, {"_id": 0}).sort([("timestamp", -1)]).limit(10))
        if not preds:
            await interaction.followup.send("No historical analyses found.")
            return
        formatted = "\n".join([f"{p.get('timestamp', 'N/A')}: {p.get('prediction', '')}" for p in preds])
        chunks = split_text(formatted)
        for chunk in chunks:
            await interaction.followup.send(chunk)
    except Exception as e:
        await interaction.followup.send(f"Error retrieving history: {e}")

@tree.command(name="news", description="Fetch the latest market news for a given symbol")
async def news(interaction: discord.Interaction, market: str, symbol: str):
    """Command to retrieve live market news for a given symbol."""
    await interaction.response.defer()
    if market.lower() == "stock":
        from predict import fetch_stock_news
        news_summary = fetch_stock_news(symbol)
    elif market.lower() == "crypto":
        from predict import fetch_crypto_news
        news_summary = fetch_crypto_news(symbol)
    else:
        await interaction.followup.send("Market type must be 'stock' or 'crypto'.")
        return
    await interaction.followup.send(news_summary)

@tree.command(name="compare", description="Compare two stocks or cryptos")
async def compare(interaction: discord.Interaction, market: str, symbol1: str, symbol2: str):
    """Command to compare two symbols."""
    await interaction.response.defer()
    if market.lower() == "stock":
        summary1, _ = analyze_stock_detailed(symbol1)
        summary2, _ = analyze_stock_detailed(symbol2)
    elif market.lower() == "crypto":
        summary1, _ = analyze_crypto(symbol1)
        summary2, _ = analyze_crypto(symbol2)
    else:
        await interaction.followup.send("Market type must be 'stock' or 'crypto'.")
        return
    comparison_text = (
        f"**{symbol1.upper()} Analysis**\n{summary1}\n\n"
        f"**{symbol2.upper()} Analysis**\n{summary2}"
    )
    chunks = split_text(comparison_text)
    for chunk in chunks:
        await interaction.followup.send(chunk)

@tree.command(name="recommend", description="Get AI recommendations for a specific stock or crypto")
async def recommend(interaction: discord.Interaction, market: str, symbol: str):
    """Command to get an AI recommendation for a given symbol."""
    await interaction.response.defer()
    recommendation = await asyncio.to_thread(get_ai_recommendation, symbol, market)
    await interaction.followup.send(recommendation)

async def send_updates():
    """Function to send periodic updates to a designated Discord channel."""
    channel = client.get_channel(DISCORD_CHANNEL_ID)
    if channel is None:
        print(f"Channel with ID {DISCORD_CHANNEL_ID} not found.")
        return

    # Get and send top 10 stock and crypto graphs.
    stock_graphs = await asyncio.to_thread(generate_top10_stock_graphs_30min)
    crypto_graphs = await asyncio.to_thread(generate_top10_crypto_graphs_30min)

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

    # Stream top 25 stock summaries as embeds.
    await channel.send("**Live Top 25 Stocks Summary (Past 30 Minutes):**")
    async for embed in async_generate_top25_stock_summaries():
        await channel.send(embed=embed)
        print("Sent one stock summary.")

    # Stream top 25 crypto summaries as embeds.
    await channel.send("**Live Top 25 Cryptos Summary (Past 30 Minutes):**")
    async for embed in async_generate_top25_crypto_summaries():
        await channel.send(embed=embed)
        print("Sent one crypto summary.")

async def periodic_updates():
    """Function to run send_updates() every 30 minutes."""
    await asyncio.sleep(1800)  # Wait 30 minutes before first update.
    while not client.is_closed():
        await send_updates()
        print("Periodic update sent.")
        await asyncio.sleep(1800)

client.run(DISCORD_TOKEN)