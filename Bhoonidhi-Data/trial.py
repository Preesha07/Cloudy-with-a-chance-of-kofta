import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()

class BhoonidhiDataFetcher:
    def __init__(self, username, password, output_dir="./raw_bhoonidhi_data"):
        self.base_url = "https://bhoonidhi-api.nrsc.gov.in"
        self.username = username
        self.password = password
        self.output_dir = output_dir
        self.token = None
        self.token_expiry = 0
        
        os.makedirs(self.output_dir, exist_ok=True)
        self._authenticate()

    def _authenticate(self):
        """Requests or refreshes the 20-minute OAuth Bearer token."""
        auth_url = f"{self.base_url}/auth/token"
        payload = {
            "userId": self.username,
            "password": self.password,
            "grant_type": "password"
        }
        try:
            response = requests.post(auth_url, json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                self.token = data.get("access_token")
                # Tokens usually expire in 1200 seconds. Buffer by 60 seconds.
                self.token_expiry = time.time() + float(data.get("expires_in", 1200)) - 60
                print("[INFO] Authentication successful. Token generated.")
            else:
                raise RuntimeError(f"Authentication rejected by server: {response.text}")
        except Exception as e:
            raise ConnectionError(f"Failed to communicate with Bhoonidhi auth node: {e}")

    def _verify_token(self):
        """Ensures token is still valid before a major transfer operation."""
        if not self.token or time.time() > self.token_expiry:
            print("[INFO] Token expired or missing. Triggering refresh sequence...")
            self._authenticate()

    def search_stac_catalog(self, bbox, start_date, end_date, collection="RESOURCESAT-2A_LISS4", limit=50):
        """
        Queries the STAC item endpoint for available satellite acquisitions.
        bbox format: [min_lon, min_lat, max_lon, max_lat]
        date format: YYYY-MM-DD
        """
        self._verify_token()
        search_url = f"{self.base_url}/data/collections/{collection}/items"
        headers = {"Authorization": f"Bearer {self.token}"}
        
        bbox_str = ",".join(map(str, bbox))
        datetime_range = f"{start_date}T00:00:00Z/{end_date}T23:59:59Z"
        
        params = {
            "bbox": bbox_str,
            "datetime": datetime_range,
            "limit": limit
        }
        
        print(f"[INFO] Scanning catalog for collection '{collection}' across target ROI...")
        response = requests.get(search_url, headers=headers, params=params, timeout=45)
        
        if response.status_code == 200:
            features = response.json().get("features", [])
            print(f"[SUCCESS] Discovered {len(features)} matching scenes in the catalog.")
            return features
        else:
            print(f"[ERROR] Catalog scan failed with code {response.status_code}: {response.text}")
            return []

    def download_scene_worker(self, item_id):
        """Worker function targeting an individual file asset download stream."""
        self._verify_token()
        download_url = f"{self.base_url}/download/{item_id}"
        headers = {"Authorization": f"Bearer {self.token}"}
        
        target_path = os.path.join(self.output_dir, f"{item_id}.zip")
        
        # Check if file already exists to avoid redundant data pulls
        if os.path.exists(target_path):
            print(f"[SKIPPED] File {item_id}.zip already exists locally.")
            return item_id, True

        print(f"[START] Initiating download stream for item: {item_id}")
        try:
            # Stream the file chunk by chunk to prevent RAM exhaustion on massive satellite scenes
            with requests.get(download_url, headers=headers, stream=True, timeout=60) as r:
                if r.status_code == 412:
                    print(f"[REJECTED] HTTP 412: Concurrency ceiling hit for {item_id}. Retrying later.")
                    return item_id, False
                r.raise_for_status()
                
                with open(target_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=65536): # 64KB chunks
                        if chunk:
                            f.write(chunk)
            print(f"[FINISH] Successfully stored {item_id}.zip")
            return item_id, True
        except Exception as e:
            print(f"[FAILURE] Transfer dropped for item {item_id}: {e}")
            if os.path.exists(target_path):
                os.remove(target_path) # Clean incomplete files
            return item_id, False

    def download_all(self, scenes):
        """Manages queue execution through a conservative thread pool."""
        item_ids = [scene['id'] for scene in scenes if 'id' in scene]
        if not item_ids:
            print("[INFO] No valid Scene IDs available to process.")
            return

        print(f"[INFO] Beginning batch transfer queue for {len(item_ids)} assets...")
        
        # Concurrency max cap bound safely to 2 to stay below the hard portal rule limit of 3
        max_threads = 2 
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            results = list(executor.map(self.download_scene_worker, item_ids))
            
        successful_downloads = sum(1 for _, success in results if success)
        print(f"\n[SUMMARY] Pipeline session closed. Stored {successful_downloads}/{len(item_ids)} archives successfully.")

# ==========================================
# RUNTIME INVOCATION
# ==========================================
if __name__ == "__main__":
    # 1. Provide your authenticated login credentials 
    BHOONIDHI_USER = os.getenv("BHOONIDHI_USER")
    BHOONIDHI_PASS = os.getenv("BHOONIDHI_PASS")
    
    # 2. Define geographical coordinates (Bounding Box) 
    # Example: Bounding Box framing portions of Northeast India (Assam / Brahmaputra basin)
    # [min_longitude, min_latitude, max_longitude, max_latitude]
    NE_INDIA_BBOX = [91.0, 25.5, 93.5, 27.0]
    
    # Initialize the fetcher pipeline
    fetcher = BhoonidhiDataFetcher(username=BHOONIDHI_USER, password=BHOONIDHI_PASS)
    
    # Step A: Fetch LISS-4 Multispectral MX23 scenes across the monsoon peak window
    liss4_scenes = fetcher.search_stac_catalog(
        bbox=NE_INDIA_BBOX,
        start_date="2026-04-30",
        end_date="2026-06-01",
        collection="RESOURCESAT-2_LISS4"
    )
    
    # Step B: Spin up download orchestration
    if liss4_scenes:
        fetcher.download_all(liss4_scenes)
        
    # Step C: Repeat for matching structural LISS-3 data layers if syncing SWIR bands
    liss3_scenes = fetcher.search_stac_catalog(
        bbox=NE_INDIA_BBOX,
        start_date="2023-06-01",
        end_date="2023-09-30",
        collection="RESOURCESAT-2A_LISS3"
    )
    if liss3_scenes:
        fetcher.download_all(liss3_scenes)