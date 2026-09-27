import { Nav } from "@/components/Nav";
import { TestSession } from "@/components/TestSession";
import { createClient } from "@/lib/supabase/server";

export const dynamic = "force-dynamic";

export default async function TestPage() {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  return (
    <>
      <Nav email={user?.email} />
      <main className="mx-auto max-w-2xl px-8 py-10">
        <TestSession />
      </main>
    </>
  );
}
