const cds = require('@sap/cds')

module.exports = cds.service.impl(async function () {
  const { ScoredResults } = this.entities

  const FASTAPI_BASE_URL = process.env.FASTAPI_BASE_URL
  if (!FASTAPI_BASE_URL) {
    throw new Error('FASTAPI_BASE_URL is not set')
  }

  this.on('READ', ScoredResults, async (req) => {
    try {
      const url = `${FASTAPI_BASE_URL}/scored-results-flat`
      console.log('READ ScoredResults triggered')
      console.log('Calling FastAPI:', url)

      const response = await fetch(url, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json'
        }
      })

      if (!response.ok) {
        throw new Error(`FastAPI request failed: ${response.status} ${response.statusText}`)
      }

      const data = await response.json()

      console.log('Rows returned from FastAPI:', data.length)

      if (!Array.isArray(data)) {
        throw new Error('FastAPI response is not an array')
      }

      return data.map((row, index) => ({
        record_id: row.record_id ?? index + 1,
        order_year: row.order_year ?? null,
        order_month: row.order_month ?? null,
        category_name: row.category_name ?? null,
        shipping_mode: row.shipping_mode ?? null,
        market: row.market ?? null,
        order_region: row.order_region ?? null,
        department: row.department ?? null,
        customer_segment: row.customer_segment ?? null,
        payment_type: row.payment_type ?? null,
        predicted_demand: row.predicted_demand ?? null,
        category_avg: row.category_avg ?? null,
        inventory_decision: row.inventory_decision ?? null,
        delay_risk: row.delay_risk ?? null,
        delay_decision: row.delay_decision ?? null,
        scored_at: row.scored_at ?? new Date().toISOString()
      }))
    } catch (err) {
      console.error('READ ScoredResults failed:', err)
      req.error(500, `Failed to fetch scored results: ${err.message}`)
    }
  })
})