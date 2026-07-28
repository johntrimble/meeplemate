import type { Page } from '@playwright/test'

/**
 * Read the persisted cache blob out of idb-keyval's store (raw IndexedDB, since
 * the app doesn't expose idb-keyval globally). The persister stores the value as
 * a serialized string under 'boardbarian-cache-v1'.
 *
 * Returns '' when absent, so callers can `expect.poll` on it to wait for the
 * async persist to land before reloading.
 */
export function readIdbCache(page: Page): Promise<string> {
  return page.evaluate(
    () =>
      new Promise<string>((resolve) => {
        const req = indexedDB.open('keyval-store')
        // Match idb-keyval's store so probing never pre-creates an incompatible DB.
        req.onupgradeneeded = () => req.result.createObjectStore('keyval')
        req.onsuccess = () => {
          let store: IDBObjectStore
          try {
            store = req.result.transaction('keyval', 'readonly').objectStore('keyval')
          } catch {
            resolve('')
            return
          }
          const g = store.get('boardbarian-cache-v1')
          g.onsuccess = () => resolve(typeof g.result === 'string' ? g.result : '')
          g.onerror = () => resolve('')
        }
        req.onerror = () => resolve('')
      }),
  )
}
