// server.js
import express from 'express';
import cors from 'cors';
import crypto from 'crypto';
import dotenv from 'dotenv';
import axios from 'axios';
import { Sequelize, DataTypes, Op } from 'sequelize';

dotenv.config();

// === Logger ===
class Logger {
  log(level, message, data = {}) {
    console.log(JSON.stringify({
      ts: new Date().toISOString(),
      level,
      msg: message,
      ...data
    }));
  }
  
  info(msg, data) { this.log('INFO', msg, data); }
  error(msg, data) { this.log('ERROR', msg, data); }
  warning(msg, data) { this.log('WARNING', msg, data); }
}

const log = new Logger();

// === CACHING LAYER ===
class VendorCache {
  constructor(ttlSeconds = 3600) {
    this._cache = new Map();
    this._timestamps = new Map();
    this._ttl = ttlSeconds * 1000; // Convert to ms
  }
  
  _generateKey(params) {
    const keyParts = Object.entries(params)
      .filter(([_, v]) => v != null)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([k, v]) => `${k}:${v}`);
    
    return crypto
      .createHash('md5')
      .update(keyParts.join('|'))
      .digest('hex')
      .substring(0, 16);
  }
  
  get(queryParams) {
    const key = this._generateKey(queryParams);
    
    if (!this._cache.has(key)) {
      return null;
    }
    
    // Check TTL
    if (Date.now() - this._timestamps.get(key) > this._ttl) {
      this._cache.delete(key);
      this._timestamps.delete(key);
      return null;
    }
    
    log.info('Cache HIT', { key });
    return this._cache.get(key);
  }
  
  set(vendorsData, queryParams) {
    const key = this._generateKey(queryParams);
    this._cache.set(key, vendorsData);
    this._timestamps.set(key, Date.now());
    log.info('Cache SET', { key, count: vendorsData.length });
    console.log('*******cache updated************');
  }
  
  invalidateAll() {
    this._cache.clear();
    this._timestamps.clear();
    log.info('Cache cleared');
  }
}

// === LLM RESPONSE CACHE ===
class LLMCache {
  constructor(ttlSeconds = 7200) {
    this._cache = new Map();
    this._timestamps = new Map();
    this._ttl = ttlSeconds * 1000;
  }
  
  _generateKey(prompt, system = null, temperature = 0.2) {
    const content = `${prompt}|${system || ''}|${temperature}`;
    return crypto
      .createHash('md5')
      .update(content)
      .digest('hex')
      .substring(0, 16);
  }
  
  get(prompt, system = null, temperature = 0.2) {
    const key = this._generateKey(prompt, system, temperature);
    
    if (!this._cache.has(key)) {
      return null;
    }
    
    if (Date.now() - this._timestamps.get(key) > this._ttl) {
      this._cache.delete(key);
      this._timestamps.delete(key);
      return null;
    }
    
    log.info('LLM Cache HIT', { key });
    return this._cache.get(key).response;
  }
  
  set(response, prompt, system = null, temperature = 0.2) {
    const key = this._generateKey(prompt, system, temperature);
    this._cache.set(key, { response });
    this._timestamps.set(key, Date.now());
    log.info('LLM Cache SET', { key });
  }
  
  invalidateAll() {
    this._cache.clear();
    this._timestamps.clear();
    log.info('LLM Cache cleared');
  }
}

// Initialize caches
const vendorCache = new VendorCache(3600);
const llmCache = new LLMCache(7200);

// === SETTINGS ===
const settings = {
  API_KEY: process.env.API_KEY || null,
  DATABASE_URL: process.env.DATABASE_URL || 'postgresql://postgres:Admin@localhost:5432/agentspace',
  OPENAI_API_KEY: process.env.OPENAI_API_KEY || null,
  OPENAI_MODEL: process.env.OPENAI_MODEL || 'gpt-3.5-turbo',
  SEED_DATA: process.env.SEED_DATA !== 'false',
  CORS_ORIGINS: process.env.CORS_ORIGINS?.split(',') || ['*'],
  RATE_LIMIT_RPS: parseFloat(process.env.RATE_LIMIT_RPS) || 5.0,
  ENABLE_VENDOR_CACHE: process.env.ENABLE_VENDOR_CACHE !== 'false',
  ENABLE_LLM_CACHE: process.env.ENABLE_LLM_CACHE !== 'false',
  DB_POOL_SIZE: parseInt(process.env.DB_POOL_SIZE) || 10,
  DB_MAX_OVERFLOW: parseInt(process.env.DB_MAX_OVERFLOW) || 20,
  LLM_TIMEOUT: parseInt(process.env.LLM_TIMEOUT) || 30000,
  PARALLEL_PROCESSING: process.env.PARALLEL_PROCESSING !== 'false',
  PORT: parseInt(process.env.PORT) || 8000
};

// === DATABASE SETUP ===
const sequelize = new Sequelize(settings.DATABASE_URL, {
  dialect: 'postgres',
  logging: false,
  pool: {
    max: settings.DB_POOL_SIZE,
    min: 0,
    acquire: 30000,
    idle: 10000
  },
  dialectOptions: {
    statement_timeout: 10000,
  }
});

// Define Vendor Model
const Vendor = sequelize.define('Vendor', {
  id: {
    type: DataTypes.INTEGER,
    primaryKey: true,
    autoIncrement: true
  },
  name: {
    type: DataTypes.STRING(255),
    allowNull: false
  },
  service_type: {
    type: DataTypes.STRING(100),
    allowNull: false
  },
  city: {
    type: DataTypes.STRING(100),
    allowNull: false
  },
  price_min: {
    type: DataTypes.INTEGER,
    allowNull: false
  },
  price_max: {
    type: DataTypes.INTEGER,
    allowNull: false
  },
  available_date: {
    type: DataTypes.DATEONLY,
    allowNull: true
  },
  contact: {
    type: DataTypes.STRING(255),
    allowNull: true
  }
}, {
  tableName: 'vendors',
  timestamps: false,
  indexes: [
    { fields: ['name'] },
    { fields: ['service_type'] },
    { fields: ['city'] },
    { fields: ['price_min'] },
    { fields: ['price_max'] },
    { fields: ['available_date'] },
    { fields: ['service_type', 'city'] },
    { fields: ['available_date', 'city'] },
    { fields: ['price_min', 'price_max'] }
  ]
});

// === OPTIMIZED LLM CLIENT ===
class OptimizedLLMClient {
  constructor() {
    this.client = axios.create({
      timeout: settings.LLM_TIMEOUT,
      headers: { 'Connection': 'keep-alive' }
    });
  }
  
  async callLLM(prompt, system = null, temperature = 0.2) {
    // Check cache first
    if (settings.ENABLE_LLM_CACHE) {
      const cachedResponse = llmCache.get(prompt, system, temperature);
      if (cachedResponse) {
        return cachedResponse;
      }
    }
    
    if (!settings.OPENAI_API_KEY) {
      const stubResponse = `[LLM-STUB] ${prompt.substring(0, 100)}...`;
      if (settings.ENABLE_LLM_CACHE) {
        llmCache.set(stubResponse, prompt, system, temperature);
      }
      return stubResponse;
    }
    
    const messages = [];
    if (system) {
      messages.push({ role: 'system', content: system });
    }
    messages.push({ role: 'user', content: prompt });
    
    try {
      const response = await this.client.post(
        'https://api.openai.com/v1/chat/completions',
        {
          model: settings.OPENAI_MODEL,
          temperature,
          max_tokens: 1000,
          messages
        },
        {
          headers: {
            'Authorization': `Bearer ${settings.OPENAI_API_KEY}`,
            'Content-Type': 'application/json'
          }
        }
      );
      
      const result = response.data.choices[0].message.content.trim();
      
      if (settings.ENABLE_LLM_CACHE) {
        llmCache.set(result, prompt, system, temperature);
      }
      
      return result;
    } catch (error) {
      if (error.code === 'ECONNABORTED') {
        log.error('LLM request timed out');
        const fallback = '[TIMEOUT] Unable to process request quickly enough';
        if (settings.ENABLE_LLM_CACHE) {
          llmCache.set(fallback, prompt, system, temperature);
        }
        return fallback;
      }
      log.error('LLM request failed', { error: error.message });
      throw error;
    }
  }
}

const llmClient = new OptimizedLLMClient();

// === FAST INTENT EXTRACTION ===
const INTENT_SYS = `You classify event-planning queries quickly. Return ONLY a JSON object (no markdown, no code blocks, no explanation) with keys: intent (QUERY_VENDORS|PLAN_EVENT|GENERAL_Q|CLARIFY|VENDOR_INFO) and slots {city,date,service_type,event_type,budget,vendor_name} where available. Use VENDOR_INFO intent when user asks about specific vendor details/contact. Dates in YYYY-MM-DD format, budget as integer. Normalize city names: bangalore/bengaluru -> Bangalore, chennai -> Chennai, mumbai -> Mumbai. Return pure JSON only, no other text.`;

const CITY_PATTERNS = {
  'chennai': 'Chennai',
  'mumbai': 'Mumbai',
  'delhi': 'Delhi',
  'bengaluru': 'Bangalore',
  'bangalore': 'Bangalore'
};

const SERVICE_PATTERNS = {
  'food': 'food',
  'camera': 'camera',
  'photography': 'camera',
  'photo': 'camera',
  'photographer': 'camera',
  'decoration': 'decoration',
  'decor': 'decoration',
  'cleaning': 'cleaning',
  'clean': 'cleaning',
  'makeup': 'makeup'
};

function fastIntentFallback(userText) {
  const t = userText.toLowerCase();
  
  let intent = 'CLARIFY';
  if (/contact|phone|details|info/.test(t)) {
    intent = 'VENDOR_INFO';
  } else if (/list|available|show|find|vendors|get|fetch/.test(t)) {
    intent = 'QUERY_VENDORS';
  } else if (/budget|price|cost/.test(t)) {
    intent = 'GENERAL_Q';
  }
  
  const slots = {};
  
  // Match city
  for (const [pattern, normalized] of Object.entries(CITY_PATTERNS)) {
    if (t.includes(pattern)) {
      slots.city = normalized;
      break;
    }
  }
  
  // Match service
  for (const [pattern, normalized] of Object.entries(SERVICE_PATTERNS)) {
    if (t.includes(pattern)) {
      slots.service_type = normalized;
      break;
    }
  }
  
  // Match date
  const dateMatch = t.match(/(\d{4}-\d{2}-\d{2})/);
  if (dateMatch) {
    slots.date = dateMatch[1];
  }
  
  // Match budget
  const budgetMatch = t.match(/(\d{4,7})/);
  if (budgetMatch) {
    slots.budget = parseInt(budgetMatch[1]);
  }
  
  return { intent, slots };
}

async function extractIntentFast(userText) {
  try {
    const raw = await llmClient.callLLM(
      `User message:\n${userText}\n\nReturn JSON only.`,
      INTENT_SYS,
      0.0
    );
    
    let jsonText = raw.trim();
    if (jsonText.startsWith('```')) {
      const start = jsonText.indexOf('{');
      const end = jsonText.lastIndexOf('}');
      if (start !== -1 && end !== -1) {
        jsonText = jsonText.substring(start, end + 1);
      }
    }
    
    return JSON.parse(jsonText);
  } catch (error) {
    log.warning('LLM intent extraction failed, using fast fallback', { error: error.message });
    return fastIntentFallback(userText);
  }
}

// === OPTIMIZED VENDOR FETCHING ===
async function fetchVendorsOptimized(slots) {
  // Normalize service type
  if (slots.service_type) {
    const serviceType = slots.service_type.toLowerCase();
    if (['photography', 'photo', 'photographer'].includes(serviceType)) {
      slots.service_type = 'camera';
    } else if (['decor', 'decorations'].includes(serviceType)) {
      slots.service_type = 'decoration';
    } else if (['clean', 'cleaner'].includes(serviceType)) {
      slots.service_type = 'cleaning';
    }
  }
  
  // Check cache first
  if (settings.ENABLE_VENDOR_CACHE) {
    const cacheKeyParams = {
      city: slots.city,
      service_type: slots.service_type,
      date: slots.date,
      vendor_name: slots.vendor_name
    };
    
    const cachedData = vendorCache.get(cacheKeyParams);
    if (cachedData) {
      log.info('Cache HIT: Found vendors', { count: cachedData.length });
      return cachedData;
    }
  }
  
  // Build query
  const where = {};
  
  if (slots.vendor_name) {
    where.name = { [Op.iLike]: `%${slots.vendor_name}%` };
  } else {
    if (slots.city) {
      where.city = slots.city;
    }
    
    if (slots.date) {
      where.available_date = slots.date;
    }
    
    if (slots.service_type) {
      where.service_type = slots.service_type;
    }
  }
  
  let vendorsList = await Vendor.findAll({ where });
  
  log.info('Database query found vendors', { count: vendorsList.length });
  
  // If no exact matches and we have date filter, try without date
  if (vendorsList.length === 0 && slots.date) {
    log.info('No exact date matches, trying without date constraint...');
    delete where.available_date;
    vendorsList = await Vendor.findAll({ where });
    log.info('Without date constraint found vendors', { count: vendorsList.length });
  }
  
  // Convert to plain objects
  const vendorsData = vendorsList.map(v => v.toJSON());
  
  // Cache the results
  if (settings.ENABLE_VENDOR_CACHE) {
    const cacheKeyParams = {
      city: slots.city,
      service_type: slots.service_type,
      date: slots.date,
      vendor_name: slots.vendor_name
    };
    vendorCache.set(vendorsData, cacheKeyParams);
  }
  
  return vendorsData;
}

// === PARALLEL PROCESSING FOR AI MATCHING ===
async function aiSemanticVendorMatchOptimized(vendors, slots) {
  if (!slots.service_type || vendors.length <= 3) {
    return vendors;
  }
  
  try {
    const vendorData = vendors.slice(0, 20).map(v => ({
      id: v.id,
      name: v.name,
      service_type: v.service_type,
      city: v.city,
      price_min: v.price_min,
      price_max: v.price_max
    }));
    
    const aiPrompt = `Match vendors to user request quickly. Return top 5 matches only.

Vendors: ${JSON.stringify(vendorData)}
User wants: ${slots.service_type}

Return JSON: {"matched_vendors": [{"vendor_id": int, "score": int}]}
No explanations, just JSON.`;
    
    const aiResponse = await llmClient.callLLM(aiPrompt, null, 0.0);
    const aiResult = JSON.parse(aiResponse.trim());
    
    const matchedData = aiResult.matched_vendors || [];
    const vendorLookup = Object.fromEntries(vendors.map(v => [v.id, v]));
    
    const resultVendors = [];
    for (const match of matchedData.slice(0, 5)) {
      if (vendorLookup[match.vendor_id]) {
        resultVendors.push(vendorLookup[match.vendor_id]);
      }
    }
    
    return resultVendors.length > 0 ? resultVendors : vendors.slice(0, 5);
  } catch (error) {
    log.warning('AI matching failed, using fallback', { error: error.message });
    return strictFallbackFilterFast(vendors, slots);
  }
}

function strictFallbackFilterFast(vendors, slots) {
  const requestedService = (slots.service_type || '').toLowerCase();
  if (!requestedService) {
    return vendors.slice(0, 5);
  }
  
  const matched = [];
  for (const vendor of vendors) {
    if (vendor.service_type.toLowerCase() === requestedService) {
      matched.push(vendor);
      if (matched.length >= 5) break;
    }
  }
  
  return matched.length > 0 ? matched : vendors.slice(0, 3);
}

// === EXPRESS APP ===
const app = express();

app.use(express.json());
app.use(cors({
  origin: settings.CORS_ORIGINS,
  credentials: true
}));

// === AUTH & RATE LIMITING ===
function apiKeyAuth(req, res, next) {
  if (settings.API_KEY && req.headers['x-api-key'] !== settings.API_KEY) {
    return res.status(401).json({ detail: 'Invalid API key' });
  }
  next();
}

const RATE_BUCKETS = new Map();

function rateGuard(req, res, next) {
  const ip = req.ip || 'unknown';
  const now = Date.now() / 1000;
  
  if (!RATE_BUCKETS.has(ip)) {
    RATE_BUCKETS.set(ip, { tokens: 5, last: now });
    return next();
  }
  
  const bucket = RATE_BUCKETS.get(ip);
  const elapsed = now - bucket.last;
  bucket.tokens = Math.min(5, bucket.tokens + elapsed * 2);
  bucket.last = now;
  
  if (bucket.tokens >= 1) {
    bucket.tokens -= 1;
    next();
  } else {
    res.status(429).json({ detail: 'Rate limit exceeded' });
  }
}

// === ENDPOINTS ===
app.get('/healthz', (req, res) => {
  res.json({
    status: 'ok',
    cache_enabled: settings.ENABLE_VENDOR_CACHE
  });
});

app.get('/cache/stats', (req, res) => {
  res.json({
    vendor_cache_size: vendorCache._cache.size,
    llm_cache_size: llmCache._cache.size,
    settings: {
      vendor_cache_enabled: settings.ENABLE_VENDOR_CACHE,
      llm_cache_enabled: settings.ENABLE_LLM_CACHE
    }
  });
});

app.post('/cache/clear', (req, res) => {
  vendorCache.invalidateAll();
  llmCache.invalidateAll();
  res.json({
    status: 'success',
    message: 'All caches cleared'
  });
});

// === OPTIMIZED CHAT ENDPOINT ===
app.post('/chat', rateGuard, apiKeyAuth, async (req, res) => {
  const startTime = Date.now();
  
  try {
    const { message } = req.body;
    
    // 1) Fast intent extraction
    const intentObj = await extractIntentFast(message);
    const slots = intentObj.slots || {};
    const intent = intentObj.intent || 'CLARIFY';
    
    // 2) Optimized vendor fetching
    const vendorsList = await fetchVendorsOptimized(slots);
    
    let reply, recommendations;
    
    // 3) Fast response generation based on intent
    if (intent === 'VENDOR_INFO' && vendorsList.length > 0) {
      const vendor = vendorsList[0];
      const contactInfo = vendor.contact 
        ? `Contact: ${vendor.contact}` 
        : 'Contact information not available';
      const availableDate = vendor.available_date 
        ? `Available: ${vendor.available_date}` 
        : 'Availability not specified';
      
      reply = `Here are the details for ${vendor.name}:\n`;
      reply += `Service: ${vendor.service_type.charAt(0).toUpperCase() + vendor.service_type.slice(1)}\n`;
      reply += `Location: ${vendor.city}\n`;
      reply += `Price Range: ₹${vendor.price_min.toLocaleString()} - ₹${vendor.price_max.toLocaleString()}\n`;
      reply += `${availableDate}\n`;
      reply += contactInfo;
      
      recommendations = [{
        vendor_id: vendor.id,
        name: vendor.name,
        service_type: vendor.service_type,
        contact: vendor.contact,
        price_range: `₹${vendor.price_min.toLocaleString()} - ₹${vendor.price_max.toLocaleString()}`
      }];
      
    } else if (intent === 'QUERY_VENDORS' && vendorsList.length > 0) {
      reply = `Found ${vendorsList.length} vendor(s):\n\n`;
      
      const summaryLines = vendorsList.slice(0, 5).map(v => 
        `• ${v.name} - ${v.service_type.charAt(0).toUpperCase() + v.service_type.slice(1)}\n` +
        `  Price: ₹${v.price_min.toLocaleString()} - ₹${v.price_max.toLocaleString()}\n` +
        `  Contact: ${v.contact || 'Not provided'}`
      );
      
      reply += summaryLines.join('\n\n');
      
      recommendations = vendorsList.slice(0, 5).map(v => ({
        vendor_id: v.id,
        name: v.name,
        service_type: v.service_type,
        contact: v.contact,
        price_range: `₹${v.price_min.toLocaleString()} - ₹${v.price_max.toLocaleString()}`
      }));
      
    } else if (vendorsList.length > 0) {
      let matchedVendors;
      if (settings.PARALLEL_PROCESSING && vendorsList.length > 5) {
        matchedVendors = await aiSemanticVendorMatchOptimized(vendorsList, slots);
      } else {
        matchedVendors = vendorsList.slice(0, 3);
      }
      
      const summary = matchedVendors.slice(0, 3).map(v =>
        `• ${v.name} (${v.service_type}) in ${v.city} – ₹${v.price_min.toLocaleString()}–₹${v.price_max.toLocaleString()}`
      ).join('\n');
      
      reply = `Here are the best options for your event:\n${summary}`;
      
      recommendations = matchedVendors.slice(0, 3).map(v => ({
        vendor_id: v.id,
        name: v.name,
        service_type: v.service_type,
        price_range: `₹${v.price_min.toLocaleString()} - ₹${v.price_max.toLocaleString()}`
      }));
      
    } else {
      reply = "I couldn't find matching vendors. Please specify city, date (YYYY-MM-DD), or service type.";
      recommendations = [];
    }
    
    const processingTime = (Date.now() - startTime) / 1000;
    
    res.json({
      reply,
      intent: intentObj,
      slots,
      recommendations,
      processing_time: Math.round(processingTime * 1000) / 1000
    });
    
  } catch (error) {
    log.error('Chat endpoint error', { error: error.message });
    const processingTime = (Date.now() - startTime) / 1000;
    
    res.json({
      reply: 'I encountered an error processing your request. Please try again.',
      intent: { intent: 'ERROR', slots: {} },
      slots: {},
      recommendations: [],
      processing_time: Math.round(processingTime * 1000) / 1000
    });
  }
});

app.get('/vendors/fast', async (req, res) => {
  const { city, service_type, limit = 10 } = req.query;
  
  const slots = {};
  if (city) slots.city = city;
  if (service_type) slots.service_type = service_type;
  
  const vendors = await fetchVendorsOptimized(slots);
  const limitedVendors = vendors.slice(0, parseInt(limit));
  
  res.json(limitedVendors);
});

app.get('/performance/metrics', (req, res) => {
  res.json({
    cache_stats: {
      vendor_cache_size: vendorCache._cache.size,
      llm_cache_size: llmCache._cache.size
    },
    settings: {
      vendor_cache_enabled: settings.ENABLE_VENDOR_CACHE,
      llm_cache_enabled: settings.ENABLE_LLM_CACHE,
      parallel_processing: settings.PARALLEL_PROCESSING,
      db_pool_size: settings.DB_POOL_SIZE,
      llm_timeout: settings.LLM_TIMEOUT
    },
    recommendations: {
      enable_caching: true,
      use_database_indexes: true,
      consider_redis: 'For production scale',
      monitor_llm_costs: true
    }
  });
});

// === STARTUP ===
async function startup() {
  try {
    // Test database connection
    await sequelize.authenticate();
    log.info('✅ PostgreSQL connected');
    
    // Sync database
    await sequelize.sync();
    
    // Seed data
    if (settings.SEED_DATA) {
      const existingCount = await Vendor.count();
      if (existingCount === 0) {
        await Vendor.bulkCreate([
          { name: 'Royal Caterers', service_type: 'food', city: 'Chennai', price_min: 15000, price_max: 60000, available_date: '2025-10-24', contact: '99999 11111' },
          { name: 'Elite Photography', service_type: 'camera', city: 'Chennai', price_min: 12000, price_max: 45000, available_date: '2025-10-24', contact: '99999 22222' },
          { name: 'Floral Decors', service_type: 'decoration', city: 'Chennai', price_min: 8000, price_max: 40000, available_date: '2025-10-24', contact: '99999 33333' },
          { name: 'Sparkle Cleaners', service_type: 'cleaning', city: 'Chennai', price_min: 3000, price_max: 12000, available_date: '2025-10-24', contact: '99999 44444' },
          { name: 'GlamUp Makeup', service_type: 'makeup', city: 'Chennai', price_min: 5000, price_max: 25000, available_date: '2025-10-24', contact: '99999 55555' },
          { name: 'Mumbai Delights', service_type: 'food', city: 'Mumbai', price_min: 20000, price_max: 80000, available_date: '2025-10-24', contact: '99999 66666' },
          { name: 'Pixel Perfect Studios', service_type: 'camera', city: 'Mumbai', price_min: 15000, price_max: 50000, available_date: '2025-10-24', contact: '99999 77777' },
          { name: 'Bloom & Blossom', service_type: 'decoration', city: 'Bangalore', price_min: 10000, price_max: 45000, available_date: '2025-10-24', contact: '99999 88888' },
          { name: 'Luxe Photography', service_type: 'camera', city: 'Delhi', price_min: 20000, price_max: 60000, available_date: '2025-10-30', contact: '99999 99999' },
          { name: 'Capital Events', service_type: 'food', city: 'Delhi', price_min: 25000, price_max: 75000, available_date: '2025-10-30', contact: '99999 10101' },
          { name: 'Delhi Decorators', service_type: 'decoration', city: 'Delhi', price_min: 15000, price_max: 50000, available_date: '2025-10-30', contact: '99999 10102' }
        ]);
        log.info('✅ Seed data added');
      }
    }
    
    // Pre-warm caches
    if (settings.ENABLE_VENDOR_CACHE) {
      log.info('🔥 Pre-warming vendor cache...');
      
      const commonQueries = [
        { city: 'Chennai' },
        { city: 'Mumbai' },
        { city: 'Delhi' },
        { service_type: 'camera' },
        { service_type: 'food' },
        { service_type: 'decoration' }
      ];
      
      for (const query of commonQueries) {
        try {
          await fetchVendorsOptimized(query);
        } catch (error) {
          log.warning('Cache pre-warming failed', { query, error: error.message });
        }
      }
      
      log.info('✅ Cache pre-warming completed');
    }
    
    log.info('🚀 Optimized service started with performance enhancements');
    
    app.listen(settings.PORT, () => {
      log.info(`Server running on port ${settings.PORT}`);
    });
    
  } catch (error) {
    log.error('❌ Startup failed', { error: error.message });
    process.exit(1);
  }
}

// === GRACEFUL SHUTDOWN ===
process.on('SIGTERM', async () => {
  log.info('SIGTERM received, shutting down gracefully...');
  await sequelize.close();
  log.info('🛑 Service shutdown completed');
  process.exit(0);
});

process.on('SIGINT', async () => {
  log.info('SIGINT received, shutting down gracefully...');
  await sequelize.close();
  log.info('🛑 Service shutdown completed');
  process.exit(0);
});

// Start the server
startup();