import { useState, useEffect, useCallback } from 'react'
import './App.css'

const DATA_URL = '/api/affine/weights/latest.json'
const METRIC_COLS = ['agentgym:webshop', 'agentgym:alfworld', 'agentgym:babyai',
  'agentgym:sciworld', 'agentgym:textcraft', 'affine:sat', 'affine:ded', 'affine:abd']
const DISPLAY_COLS = ['#', 'UID', 'Model', 'Rev', ...METRIC_COLS, 'Score', 'Pts', 'Elig']

function App() {
  const [data, setData] = useState(null)
  const [sortCol, setSortCol] = useState('Score')
  const [sortAsc, setSortAsc] = useState(false)
  const [selectedMiner, setSelectedMiner] = useState(null)
  const [countdown, setCountdown] = useState(60)
  const [modelFilter, setModelFilter] = useState('')

  const fetchData = useCallback(async () => {
    try {
      const res = await fetch(DATA_URL + '?t=' + Date.now())
      const json = await res.json()
      setData(json)
    } catch (e) {
      console.error('Fetch error:', e)
    }
  }, [])

  useEffect(() => {
    fetchData()
  }, [fetchData])

  useEffect(() => {
    const timer = setInterval(() => {
      setCountdown(prev => {
        if (prev <= 1) {
          fetchData()
          return 60
        }
        return prev - 1
      })
    }, 1000)
    return () => clearInterval(timer)
  }, [fetchData])

  const handleSort = (colName) => {
    if (colName === '#') return
    if (sortCol === colName) {
      setSortAsc(!sortAsc)
    } else {
      setSortCol(colName)
      setSortAsc(false)
    }
  }

  const parseMetricValue = (val) => {
    if (val === null || val === undefined) return 0
    if (typeof val === 'number') return val
    if (typeof val === 'string') {
      let cleaned = val.replace(/^\*|\*$/g, '')
      if (cleaned.includes('/')) {
        const num = parseFloat(cleaned.split('/')[0])
        if (!isNaN(num)) return num
      }
      const num = parseFloat(cleaned)
      if (!isNaN(num)) return num
    }
    return 0
  }

  const calculateScore = (row) => {
    const metricIndices = METRIC_COLS.map(c => data.data.header.indexOf(c))
    const values = metricIndices.map(i => parseMetricValue(row[i]))
    const avg = values.reduce((a, b) => a + b, 0) / values.length
    return avg.toFixed(2)
  }

  const getRowValue = (row, col) => {
    if (col === '#') return 0
    if (col === 'Score') return parseFloat(calculateScore(row))
    const idx = data.data.header.indexOf(col)
    if (idx === -1) return -Infinity
    return parseMetricValue(row[idx])
  }

  const getSortedRows = () => {
    if (!data) return []
    let rows = [...data.data.rows]

    // Filter by model name
    if (modelFilter) {
      const modelIdx = data.data.header.indexOf('Model')
      rows = rows.filter(row =>
        row[modelIdx]?.toLowerCase().includes(modelFilter.toLowerCase())
      )
    }

    if (sortCol !== null && sortCol !== '#') {
      rows.sort((a, b) => {
        let va = getRowValue(a, sortCol)
        let vb = getRowValue(b, sortCol)

        if (typeof va === 'string' && typeof vb === 'string') {
          return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va)
        }
        return sortAsc ? va - vb : vb - va
      })
    }
    return rows
  }

  const formatCell = (val, colName) => {
    if (val === null || val === undefined) return val
    let display = val

    // Remove confidence interval from metrics (e.g., "98.22/[97.32,98.85]/500" -> "98.22/500")
    if (typeof display === 'string' && display.includes('/[')) {
      display = display.replace(/\/\[[^\]]+\]\//, '/')
    }

    // Handle winner markers
    if (typeof display === 'string' && display.startsWith('*') && display.endsWith('*')) {
      return <span className="winner">{display.slice(1, -1)}</span>
    }

    // Truncate Model column
    if (colName === 'Model' && typeof display === 'string' && display.length > 30) {
      return <span title={display}>{display.slice(0, 30)}...</span>
    }

    return display
  }

  if (!data) return <div className="dashboard"><h1>Loading...</h1></div>

  const hotkeyIndex = data.data.header.indexOf('Hotkey')
  const eligIndex = data.data.header.indexOf('Elig')

  const getCellValue = (row, colName, rowIndex) => {
    if (colName === '#') return rowIndex + 1
    if (colName === 'Score') return calculateScore(row)
    const idx = data.data.header.indexOf(colName)
    return idx !== -1 ? row[idx] : ''
  }

  return (
    <div className="dashboard">
      <h1>Affine Leaderboard</h1>
      <div className="info">
        {new Date(data.timestamp * 1000).toLocaleString()} | Block: {data.block} |
        Eligible: {data.data.stats.eligible_count} | Active: {data.data.stats.active_count}
      </div>
      <div className="refresh-timer">Next refresh in: {countdown}s</div>

      <div className="filter-container">
        <input
          type="text"
          placeholder="Filter by model name..."
          value={modelFilter}
          onChange={(e) => setModelFilter(e.target.value)}
          className="model-filter"
        />
      </div>

      <div className="table-container">
        <table>
          <thead>
            <tr>
              {DISPLAY_COLS.map((col) => (
                <th
                  key={col}
                  onClick={() => handleSort(col)}
                  className={sortCol === col ? (sortAsc ? 'sorted-asc' : 'sorted-desc') : ''}
                >
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {getSortedRows().map((row, idx) => (
              <tr
                key={idx}
                className={row[eligIndex] === 'Y' ? 'eligible-y' : ''}
                onClick={() => setSelectedMiner(data.data.miners[row[hotkeyIndex]])}
              >
                {DISPLAY_COLS.map((colName, i) => (
                  <td key={i}>{formatCell(getCellValue(row, colName, idx), colName)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {selectedMiner && (
        <Modal miner={selectedMiner} onClose={() => setSelectedMiner(null)} />
      )}
    </div>
  )
}

function Modal({ miner, onClose }) {
  return (
    <div className="modal-overlay" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal-content">
        <button className="modal-close" onClick={onClose}>&times;</button>
        <h2 className="modal-title">{miner.model}</h2>
        <div className="detail-grid">
          <div className="detail-card">
            <h3>Basic Info</h3>
            <p><span className="label">UID:</span> {miner.uid}</p>
            <p><span className="label">Hotkey:</span> {miner.hotkey}</p>
            <p><span className="label">Revision:</span> {miner.revision}</p>
            <p><span className="label">Total Score:</span> {miner.total_score}</p>
            <p><span className="label">Weight:</span> {miner.weight}</p>
            <p><span className="label">Eligible:</span> {miner.eligible ? 'Yes' : 'No'}</p>
            <p><span className="label">First Block:</span> {miner.first_block}</p>
          </div>
          <div className="detail-card">
            <h3>Layer Points</h3>
            {Object.entries(miner.layer_points).map(([k, v]) => (
              <p key={k}><span className="label">{k}:</span> {v}</p>
            ))}
          </div>
          {Object.entries(miner.environments).map(([env, d]) => (
            <div className="detail-card" key={env}>
              <h3>{env} {d.is_winner ? '🏆' : ''}</h3>
              <p><span className="label">Accuracy:</span> {(d.accuracy * 100).toFixed(2)}%</p>
              <p><span className="label">Count:</span> {d.count}</p>
              <p><span className="label">CI:</span> [{(d.confidence_interval.lower * 100).toFixed(2)}%, {(d.confidence_interval.upper * 100).toFixed(2)}%]</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default App
