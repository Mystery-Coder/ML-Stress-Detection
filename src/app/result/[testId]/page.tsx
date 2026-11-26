import { Nav } from "@/components/Nav";
import { UnlockResults } from "@/components/UnlockResults";
import { createClient } from "@/lib/supabase/server";

export const dynamic = "force-dynamic";

export default async function ResultPage({ params }: { params: Promise<{ testId: string }> }) {
  const { testId } = await params;
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  return (
    <>
      <Nav email={user?.email} />
      <main className="mx-auto w-full max-w-5xl px-6 py-10 sm:px-8 lg:px-10">
        <UnlockResults testId={testId} />
      </main>
    </>
  );
}
