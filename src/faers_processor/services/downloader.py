"""Service for downloading FAERS data."""
import asyncio
import aiohttp
import aiofiles
from bs4 import BeautifulSoup
from pathlib import Path
import zipfile
from typing import List, Set
import logging
from abc import ABC, abstractmethod

class DataDownloader(ABC):
    """Abstract base class for data downloading."""
    
    @abstractmethod
    async def download_file(self, url: str, destination: Path) -> None:
        """Download a single file."""
        pass
    
    @abstractmethod
    async def extract_file(self, zip_path: Path, extract_path: Path) -> None:
        """Extract a downloaded zip file."""
        pass

class FAERSDownloader(DataDownloader):
    """FAERS specific data downloader implementation."""
    
    FAERS_URL = "https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html"
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def get_download_links(self) -> Set[str]:
        """Get all FAERS download links."""
        async with self.session.get(self.FAERS_URL) as response:
            soup = BeautifulSoup(await response.text(), 'lxml')
            links = {a['href'] for a in soup.find_all('a', href=True)
                    if '.zip' in a['href'] and 'ascii' in a['href']}
            return links
            
    async def download_file(self, url: str, destination: Path) -> None:
        """Download a single FAERS file."""
        async with self.session.get(url) as response:
            if response.status == 200:
                async with aiofiles.open(destination, 'wb') as f:
                    await f.write(await response.read())
            else:
                logging.error(f"Failed to download {url}: {response.status}")
                
    async def extract_file(self, zip_path: Path, extract_path: Path) -> None:
        """Extract a downloaded zip file."""
        extract_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        zip_path.unlink()  # Remove zip file after extraction
        
    async def download_all(self):
        """Download and extract all FAERS data."""
        links = await self.get_download_links()
        for link in links:
            zip_name = self.output_dir / Path(link).name
            extract_dir = self.output_dir / zip_name.stem
            
            await self.download_file(link, zip_name)
            await self.extract_file(zip_name, extract_dir)
            
            # Handle special case for 2018Q1
            if '2018q1' in str(extract_dir):
                demo_file = extract_dir / 'ascii' / 'DEMO18Q1_new.txt'
                if demo_file.exists():
                    demo_file.rename(demo_file.parent / 'DEMO18Q1.txt')
