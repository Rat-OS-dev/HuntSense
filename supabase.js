import { createClient } from "https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2.49.1/+esm";
const SUPABASE_URL = "https://rtyzudbbwoulrysvypql.supabase.co";
const SUPABASE_PUBLISHABLE_KEY = "sb_publishable_kbK6j2X2JxNs9YE8VSBaSw_3co8Kh8C";
export const supabase = createClient(SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY);