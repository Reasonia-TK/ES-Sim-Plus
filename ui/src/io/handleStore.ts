// ブラウザの FileSystemFileHandle を IndexedDB に保存する (上書き保存・最近使ったファイルから開くため)。

const DB_NAME = "es-sim-ui";
const STORE = "file-handles";

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function withStore<T>(mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  const db = await openDb();
  try {
    return await new Promise<T>((resolve, reject) => {
      const tx = db.transaction(STORE, mode);
      const req = fn(tx.objectStore(STORE));
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  } finally {
    db.close();
  }
}

export async function putHandle(handle: FileSystemFileHandle): Promise<string> {
  const id = crypto.randomUUID();
  await withStore("readwrite", (s) => s.put(handle, id));
  return id;
}

export async function getHandle(id: string): Promise<FileSystemFileHandle | undefined> {
  return withStore<FileSystemFileHandle | undefined>("readonly", (s) => s.get(id));
}

export async function deleteHandle(id: string): Promise<void> {
  await withStore("readwrite", (s) => s.delete(id));
}
