import { createServerClient } from "@supabase/ssr";
import { NextResponse, type NextRequest } from "next/server";
import { supabasePublishableKey, supabaseUrl } from "@/lib/supabase/env";

export async function proxy(request: NextRequest) {
	let response = NextResponse.next({ request });
	const url = supabaseUrl();
	const anon = supabasePublishableKey();
	if (!url || !anon) return response;

	const supabase = createServerClient(url, anon, {
		cookies: {
			getAll() {
				return request.cookies.getAll();
			},
			setAll(all) {
				all.forEach(({ name, value }) =>
					request.cookies.set(name, value),
				);
				response = NextResponse.next({ request });
				all.forEach(({ name, value, options }) =>
					response.cookies.set(name, value, options),
				);
			},
		},
	});
	const { data } = await supabase.auth.getUser();
	const path = request.nextUrl.pathname;
	const isAuthPage =
		path.startsWith("/login") || path.startsWith("/register");
	if (!data.user && !isAuthPage && path !== "/") {
		return NextResponse.redirect(new URL("/login", request.url));
	}
	if (data.user && isAuthPage) {
		return NextResponse.redirect(new URL("/dashboard", request.url));
	}
	return response;
}

export const config = {
	matcher: ["/login", "/register", "/dashboard", "/test", "/result/:path*"],
};
